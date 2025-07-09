from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.documents.extend(texts)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]
