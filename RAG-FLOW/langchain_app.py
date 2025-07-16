import os
import faiss
import streamlit as st
import pickle
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --------------- CONFIG ---------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"
DOCS_PATH = "documents.pkl"

# --------------- SAMPLE DATA ------------ #
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Vector Search",
        "content": "Vector search uses embeddings to find semantically similar documents. FAISS and Pinecone are common tools."
    },
    {
        "id": "doc2",
        "title": "Language Models",
        "content": "Language models like GPT-4 generate responses based on input prompts. They’re used in chatbots and assistants."
    },
    {
        "id": "doc3",
        "title": "Embeddings",
        "content": "Embeddings are vector representations of text, used for semantic search, clustering, and retrieval systems."
    },
]

# --------------- EMBEDDINGS -------------- #
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# --------------- DOCUMENT PREP ----------- #
def build_vectorstore(documents, persist=True):
    texts = [doc["content"] for doc in documents]
    metadatas = [{"id": doc["id"], "title": doc["title"]} for doc in documents]
    embeddings = get_embedding_model()
    db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    if persist:
        db.save_local(INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)
    return db

# --------------- LOAD FAISS OR BUILD ----- #
def load_or_build_vectorstore():
    if os.path.exists(f"{INDEX_PATH}/index.faiss"):
        embeddings = get_embedding_model()
        return FAISS.load_local(INDEX_PATH, embeddings)
    else:
        return build_vectorstore(DOCUMENTS)

# --------------- QA Chain ---------------- #
def build_qa_chain(vectorstore):
    prompt_template = """Use the following context to answer the question.
    If the answer isn't in the context, say "I don't know."

    Context:
    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff", chain_type_kwargs={"prompt": prompt})

# --------------- STREAMLIT UI ------------ #
def run_streamlit_app():
    st.set_page_config(page_title="RAG QA", layout="centered")
    st.title("📚 Retrieval-Augmented Q&A with LangChain + OpenAI")

    query = st.text_input("Ask a question:", placeholder="e.g. What is vector search?")
    vectorstore = load_or_build_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.run(query)
        st.success(result)

        with st.expander("🔎 Retrieved Documents"):
            docs = vectorstore.similarity_search(query, k=3)
            for doc in docs:
                st.markdown(f"- **{doc.metadata['title']}**: {doc.page_content[:200]}...")

# --------------- MAIN -------------------- #
if __name__ == "__main__":
    run_streamlit_app()
