# CSR Process Chatbot (Free Model, API + RAG)

A minimal Flask app that lets your CSRs chat with a bot grounded in your process docs.  
- Uses **local embeddings** (free) via `sentence-transformers` + **FAISS** for retrieval.  
- Uses **Groq** API (free tier) for fast Llama 3.1 inference.  
- Upload PDFs/DOCX/TXT/MD and click **Rebuild Index** to make them queryable.

## 1) Setup

```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set your Groq API key (free: https://console.groq.com/)
# Windows PowerShell:
$env:GROQ_API_KEY="paste-your-key"
# macOS/Linux:
export GROQ_API_KEY="paste-your-key"
```

Optional env vars:
```
GROQ_MODEL=llama-3.1-8b-instant
TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=120
KNOWLEDGE_DIR=knowledgebase
STORAGE_DIR=storage
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
PORT=5000
```

## 2) Add Documents
Put your PDFs/DOCX/TXT/MD into the `knowledgebase/` folder.  
Or use the **Upload Doc** form in the UI.

Then click **Rebuild Index** (button on the top right).

## 3) Run the App
```bash
python app.py
# open http://localhost:5000
```

## 4) How It Works
- **RAG** pipeline: your question → embed → retrieve top-K chunks (FAISS) → prompt Llama with those chunks.
- Answers are constrained to the provided context. If not found, the bot says it doesn't know.

## 5) Deploy (quick options)
- **Render** or **Railway**: push this repo and set env var `GROQ_API_KEY`; add a persistent volume for `storage/` if you want the index to persist.
- **Docker (local)**:
  ```bash
  docker build -t csr-rag .
  docker run -p 5000:5000 -e GROQ_API_KEY=xxx csr-rag
  ```

## 6) Notes
- This is *not* fine-tuning; it’s retrieval + prompting so you don’t leak private data to training.
- You can swap Groq for any OpenAI-compatible API easily in `call_groq()`.
- If you prefer fully offline, swap the model call to a local server (e.g., Ollama) and change `call_groq()` endpoint.
