from retriever import Retriever
from generator import Generator

# Sample documents
docs = [
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
    "France is in Europe and known for its wine and cheese."
]

retriever = Retriever(docs)
generator = Generator()

def rag(query):
    context_docs = retriever.get_relevant_docs(query)
    context = " ".join(context_docs)
    return generator.generate(context, query)

if __name__ == "__main__":
    user_query = input("Ask a question: ")
    print("Answer:", rag(user_query))
