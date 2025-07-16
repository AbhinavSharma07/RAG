import faiss
import json
import textwrap
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "gpt2"
CHUNK_SIZE = 300
TOP_K = 3


# -----------------------------
# Sample Documents
# -----------------------------
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Introduction to Vector Search",
        "content": (
            "Vector search is a technique that uses dense embeddings to find semantically similar documents. "
            "It replaces traditional keyword search by mapping queries and documents into a shared vector space. "
            "Libraries like FAISS, Pinecone, and Weaviate are commonly used for vector search in modern NLP pipelines."
        ),
    },
    {
        "id": "doc2",
        "title": "What are Language Models?",
        "content": (
            "Language models are trained to predict the next token in a sequence. "
            "Large language models like GPT-3 or GPT-4 can generate coherent text, answer questions, and perform tasks by conditioning on a prompt. "
            "They are often combined with retrieval systems to provide grounded and accurate answers."
        ),
    },
    {
        "id": "doc3",
        "title": "Understanding Embeddings",
        "content": (
            "Embeddings are numerical representations of text. Sentence-transformers provide high-quality embeddings "
            "that can be used for clustering, similarity search, and retrieval-augmented generation (RAG)."
        ),
    },
]


# -----------------------------
# Chunking Utility
# -----------------------------
class Chunker:
    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> List[str]:
        return textwrap.wrap(text, width=self.chunk_size)


# -----------------------------
# Embedding Model Wrapper
# -----------------------------
class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_tensor=False))


# -----------------------------
# Indexer and Retriever
# -----------------------------
class FAISSRetriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.index_to_doc = {}

    def build_index(self, documents: List[Dict]) -> None:
        contents = [doc["content"] for doc in documents]
        vectors = self.embedding_model.encode(contents)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

        self.index_to_doc = {i: doc for i, doc in enumerate(documents)}
        print(f"[INFO] Indexed {len(documents)} chunks.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        query_vec = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for idx in indices[0]:
            if idx in self.index_to_doc:
                results.append(self.index_to_doc[idx])
        return results


# -----------------------------
# Text Generator
# -----------------------------
class Generator:
    def __init__(self, model_name: str = GENERATOR_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, query: str, context_docs: List[Dict]) -> str:
        context = "\n".join([doc["content"] for doc in context_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------------
# Full RAG Pipeline
# -----------------------------
class RAGPipeline:
    def __init__(self):
        self.chunker = Chunker(CHUNK_SIZE)
        self.embedding_model = EmbeddingModel()
        self.retriever = FAISSRetriever(self.embedding_model)
        self.generator = Generator()

    def prepare(self, raw_documents: List[Dict]) -> None:
        chunked_docs = []
        for doc in raw_documents:
            chunks = self.chunker.chunk_text(doc["content"])
            for chunk in chunks:
                chunked_docs.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": chunk
                })
        self.retriever.build_index(chunked_docs)

    def query(self, query: str) -> str:
        retrieved_docs = self.retriever.retrieve(query)
        return self.generator.generate(query, retrieved_docs)


# -----------------------------
# Main CLI Loop
# -----------------------------
def main():
    print("🔍 Initializing RAG system...")
    rag = RAGPipeline()
    rag.prepare(DOCUMENTS)

    print("✅ RAG system ready! Type your questions below.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag.query(query)
        print(f"\n💬 RAG Answer:\n{answer}")


if __name__ == "__main__":
    main()
