from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
nltk.download("punkt")

class MultiVectorRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.sentences = []
        self.doc_map = []
        self.index = None

    def add_documents(self, docs):
        all_sents = []
        for i, doc in enumerate(docs):
            sents = nltk.sent_tokenize(doc)
            self.sentences.extend(sents)
            self.doc_map.extend([i] * len(sents))
            all_sents.extend(sents)

        embeddings = self.model.encode(all_sents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.docs = docs

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_vec, top_k)
        doc_ids = {self.doc_map[i] for i in indices[0]}
        return [self.docs[i] for i in doc_ids]

if __name__ == "__main__":
    docs = [
        "Paris is the capital of France. It is a city of art.",
        "The Eiffel Tower is located in Paris and is a famous landmark.",
        "France is in Europe. It's famous for wine and cheese."
    ]
    rag = MultiVectorRAG()
    rag.add_documents(docs)
    print("Retrieved:", rag.retrieve("Where is the Eiffel Tower?"))
