# DocuMind AI

An AI-powered PDF assistant. Upload a document, ask questions about it, and get context-aware answers with citations to the source passages.

> Status: In progress. Backend scaffolding is in place. Frontend and the full RAG pipeline are being built.

## Planned stack

- **Frontend:** Angular
- **Backend:** Python, FastAPI
- **RAG:** LangChain, Groq LLM (Qwen), Hugging Face embeddings
- **Vector store:** In-memory for now; Chroma or FAISS for persistence

## How it works

1. Upload a PDF; pypdf extracts the text.
2. The text is split into chunks and embedded with a Hugging Face model.
3. Chunks are stored in a vector store.
4. A question is embedded and the most relevant chunks are retrieved.
5. Groq's Qwen model answers using only those chunks and cites them.

## Project structure

    backend/
    ├── app/
    │   ├── main.py        # FastAPI entry point
    │   └── services/      # ingestion, retrieval, and answer generation
    ├── data/              # uploaded documents
    └── requirements.txt

## Setup (backend)

    cd backend
    python -m venv .venv
    source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp app/.env_example app/.env   # add your GROQ_API_KEY
    uvicorn app.main:app --reload

## Roadmap

- [x] Project structure and dependencies
- [ ] PDF ingestion and chunking
- [ ] Embedding and in-memory retrieval
- [ ] Answer generation with source citations
- [ ] Angular frontend for upload and chat
- [ ] Persistent vector store (Chroma or FAISS)