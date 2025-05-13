# Run the app using: python -m streamlit run "c:\Users\Yupen\Desktop\RAG\ResumeGPT.py"

import os
from dotenv import load_dotenv  # Load environment variables from .env
import fitz  # PyMuPDF, used to extract text from PDF
import faiss  # Facebook AI Similarity Search - used for fast vector retrieval
import numpy as np
import requests
import streamlit as st
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer
from typing import List

# ==================== Load API Key from .env ====================
load_dotenv()
GMI_API_URL = "https://api.gmi-serving.com/v1/chat/completions"
GMI_API_KEY = os.getenv("GMI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # A fast, general-purpose sentence embedding model


# ==================== PDF Chunking ====================
def extract_chunks_from_file(uploaded_file, chunk_size: int = 300) -> List[str]:
    """
    Extract text from .pdf, .docx, or .txt file and split into chunks.

    Args:
        uploaded_file: File-like object from Streamlit uploader.
        chunk_size: Approximate words per chunk.

    Returns:
        List[str]: Chunked resume text.
    """
    import io

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".pdf"):
            # Use fitz to extract PDF text
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)

        elif file_name.endswith(".docx"):
            from docx import Document

            docx_doc = Document(uploaded_file)
            text = "\n".join(p.text for p in docx_doc.paragraphs if p.text.strip())

        elif file_name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")

        else:
            return [
                "❌ Unsupported file format. Only PDF, DOCX, and TXT are supported."
            ]

        words = text.split()
        if len(words) <= 1000:
            return [text]

        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    except Exception as e:
        return [f"❌ Error processing file: {str(e)}"]


# ==================== Load Sentence Embedding Model ====================
@st.cache_resource
def load_embedding_model():
    # Loads the MiniLM embedding model and caches it across Streamlit reruns
    return SentenceTransformer(EMBEDDING_MODEL)


# ==================== Convert Chunks to Vectors ====================
def embed_chunks(chunks: List[str], model) -> np.ndarray:
    # Converts text chunks to 384-dimensional float32 vectors
    embeddings = model.encode(chunks, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


# ==================== Build FAISS Index for Retrieval ====================
@st.cache_resource
def build_faiss_index(embeddings: np.ndarray):
    # Creates a FAISS index using L2 distance and adds the embedding vectors to it
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# ==================== Retrieve Top-K Relevant Chunks ====================
def retrieve_top_k(query: str, model, chunks, index, k: int = 3) -> List[str]:
    """
    Takes a user query, embeds it, and retrieves the top-K most relevant resume chunks using FAISS.
    Automatically adjusts k based on the actual number of chunks available.

    Args:
        query (str): The user's question.
        model: The embedding model.
        chunks: Original text chunks.
        index: FAISS index containing embedded vectors.
        k (int): Number of top chunks to retrieve.

    Returns:
        List[str]: Top-k relevant chunks based on vector similarity.
    """
    query_embedding = model.encode([query])

    # Adjust k to not exceed the actual number of chunks
    k = min(k, len(chunks))

    D, I = index.search(np.array(query_embedding, dtype="float32"), k)

    return [chunks[i] for i in I[0]]


# ==================== Call GMI API for RAG-style Answering ====================
def call_gmi_rag(query: str, context: List[str]) -> str:
    """
    Sends the user query and relevant resume chunks to GMI Cloud’s LLM API, and returns the generated answer.

    Args:
        query (str): The user's question.
        context (List[str]): Retrieved relevant resume chunks.

    Returns:
        str: LLM's answer based on the provided context.
    """
    if not GMI_API_KEY:
        return "❌ GMI_API_KEY not set. Please check your .env file."

    # Chat messages using OpenAI-compatible format
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the resume excerpts to answer the question.",
        },
        {
            "role": "user",
            "content": f"{query}\n\nResume Context:\n{chr(10).join(context)}",
        },
    ]

    # Set model + output token limit (2048 = allows long answers)
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": messages,
        "temperature": 0.0,  # deterministic output
        "max_tokens": 2048,  # limits the maximum length of the response
    }

    headers = {
        "Authorization": f"Bearer {GMI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Make the POST request to GMI API
    resp = requests.post(GMI_API_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        return f"❌ API Error {resp.status_code}: {resp.text}"

    # Extract the model’s response content
    return (
        resp.json()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "No response content.")
    )


# ==================== Streamlit Web App ====================
st.set_page_config(page_title="ResumeChatAI", layout="wide")
st.title("📄 ResumeChatAI: Understand any resume through conversation")

# File upload widget
uploaded_file = st.file_uploader(
    "Select a resume document", type=["pdf", "docx", "txt"]
)

# Main processing pipeline
if uploaded_file:
    st.success("✅ Resume uploaded and processing...")

    # Step 1: Extract & split resume
    chunks = extract_chunks_from_file(uploaded_file)

    if len(chunks) == 1 and chunks[0].startswith("❌"):
        st.error(chunks[0])
        st.stop()

    # Step 2: Load embedding model
    embed_model = load_embedding_model()

    # Step 3: Convert text chunks to embeddings
    embeddings = embed_chunks(chunks, embed_model)

    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 5: Ask user question
    query = st.text_input("Ask a question about the resume:")

    if query:
        top_chunks = retrieve_top_k(query, embed_model, chunks, index)

        with st.expander("Retrieved Resume Chunks"):
            for i, chunk in enumerate(top_chunks, 1):
                st.write(f"**Chunk {i}:** {chunk[:200]}...")

        answer = call_gmi_rag(query, top_chunks)
        st.markdown("### 🤖 AI Response")
        st.info(answer)
