
# 🤖 ResumeChatAI

**Understand any resume through conversation.**  
Upload a PDF, DOCX, or TXT resume, and ask questions to extract insights using RAG and LLM.

---

## 🚀 Features

- 📄 Upload PDF, DOCX, or TXT resume files
- 🧠 Sentence embeddings with `all-MiniLM-L6-v2`
- ⚡ Fast semantic search via FAISS
- 🤖 AI answers generated using GMI Cloud LLM (DeepSeek-R1)
- 🧪 Streamlit-based front-end, no backend required

---

## 🔧 Setup

1. **Clone this repository**  
```bash
git clone https://github.com/YOUR_USERNAME/ResumeChatAI.git
cd ResumeChatAI
```

2. **Install Python dependencies**  
```bash
pip install -r requirements.txt
```

3. **Add your GMI API Key**  
Create a `.env` file in the root directory and add:

```
GMI_API_KEY=your_actual_gmi_api_key
```

You can refer to `.env.example`.

4. **Run the Streamlit app**  
```bash
python -m streamlit run ResumeChatAI.py
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New App", select your GitHub repo, and set the file path to `ResumeChatAI.py`
4. In **Secrets**, add your API key:
   ```
   GMI_API_KEY = your_actual_key
   ```

---

## 📁 Project Structure

```
ResumeChatAI/
├── ResumeChatAI.py         # Streamlit App
├── requirements.txt        # Dependencies
├── .env.example            # Environment variable template
└── README.md               # Project overview
```

---

## 🛡️ Disclaimer

This app uses external LLM APIs. Do not upload resumes with sensitive or personal data unless appropriate consent and security measures are in place.

---

## 📄 License

MIT License © 2025 Yupeng Tang
