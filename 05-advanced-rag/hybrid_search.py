from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class HybridRetriever:
    def __init__(self, docs, model_name="all-MiniLM-L6-v2"):
        self.docs = docs
        self.tokenized_docs = [doc.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(docs, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=3, alpha=0.5):
        bm25_scores = self.bm25.get_scores(query.lower().split())
        query_vec = self.model.encode([query], convert_to_numpy=True)
        _, dense_indices = self.index.search(query_vec, len(self.docs))
        dense_scores = np.zeros(len(self.docs))
        for i, idx in enumerate(dense_indices[0]):
            dense_scores[idx] = 1 / (1 + i)  # inverse rank

        hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores
        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [self.docs[i] for i in ranked_indices]

if __name__ == "__main__":
    docs = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "France is in Europe and known for its wine."
    ]
    retriever = HybridRetriever(docs)
    results = retriever.retrieve("Where is Eiffel Tower?")
    print("Retrieved docs:\n", results)
