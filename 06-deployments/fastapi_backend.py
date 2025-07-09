from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI(title="RAG Backend API")

# Sample corpus
docs = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "France is known for cheese and wine."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

class Query(BaseModel):
    question: str

@app.post("/query/")
def get_answer(query: Query):
    question = query.question
    query_vec = model.encode([question], convert_to_numpy=True)
    _, I = index.search(query_vec, 2)
    context = " ".join([docs[i] for i in I[0]])
    return {
        "question": question,
        "context": context,
        "answer": context.split('.')[0]
    }

# Run with: uvicorn fastapi_backend:app --reload