from vector_store import VectorStore

class Retriever:
    def __init__(self, documents):
        self.vs = VectorStore()
        self.vs.add_documents(documents)

    def get_relevant_docs(self, query, top_k=3):
        return self.vs.retrieve(query, top_k)
