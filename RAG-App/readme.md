# 🔍 Advanced Full-Stack RAG (Retrieval-Augmented Generation) Application

This is a production-grade, full-stack RAG application that allows users to upload documents (PDFs), ask natural language questions, and get answers grounded in those documents — powered by OpenAI, FAISS, ElasticSearch, and a React-based chat UI.

---

## 🚀 Features

- 📄 PDF document upload and ingestion
- 🧠 OpenAI Embeddings (`text-embedding-3-small`)
- 🔍 Hybrid Retrieval:
  - 🔹 Vector similarity search (FAISS)
  - 🔹 Keyword/BM25 search (ElasticSearch)
- 🧩 Chunk-level citations in answers
- 💬 Chat UI with token-by-token streaming (WebSocket)
- 🐳 Fully Dockerized: backend, frontend, ElasticSearch, Grafana
- 📊 Monitoring with Prometheus and Grafana
- 🧪 Ready for CI/CD, local or cloud deployment
- 🔐 Environment-variable based configuration
- 🖥️ Optional local LLM support (via Ollama)

---

## 🛠️ Tech Stack

| Layer      | Tech                             |
|------------|----------------------------------|
| Backend    | FastAPI + Python                 |
| Embeddings | OpenAI API (fallback: HF models) |
| Retrieval  | FAISS + ElasticSearch (BM25)     |
| Streaming  | WebSockets + Token Streaming     |
| DevOps     | Docker, Docker Compose           |
| Monitoring | Prometheus + Grafana             |

---

## 🐳 Build and run with Docker

```bash
docker-compose up --build
```

---

Frontend: http://localhost:3000  
Backend: http://localhost:8000  
ElasticSearch: http://localhost:9200  
Grafana: http://localhost:3001 (login: admin / admin)
