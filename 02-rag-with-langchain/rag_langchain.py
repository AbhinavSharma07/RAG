from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
docs = ["Paris is the capital of France.", 
        "Eiffel Tower is in Paris.", 
        "France is famous for wine."]

# Save docs temporarily
with open("docs.txt", "w") as f:
    for line in docs:
        f.write(f"{line}\n")

loader = TextLoader("docs.txt")
documents = loader.load()

# Embedding + Vector store
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embedding)

# Set up Retrieval-QA Chain
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Ask a question
question = "Where is the Eiffel Tower?"
answer = qa.run(question)
print("Q:", question)
print("A:", answer)
