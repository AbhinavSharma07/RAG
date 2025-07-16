import os
import faiss
import streamlit as st
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List

# ----------------------
# Configuration
# ----------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"
DOCS_PATH = "documents.pkl"

# Make sure to set your OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # or set directly as a string here

# ----------------------
# Sample Document Data
# ----------------------
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Vector Search",
        "content": "Vector search uses embeddings to find semantically similar documents. FAISS and Pinecone are commonly used tools."
    },
    {
        "id": "doc2",
        "title": "Language Models",
        "content": "Language models like GPT-4 generate responses based on prompts. They can perform question answering, summarization, and more."
    },
    {
        "id": "doc3",
        "title": "Embeddings",
        "content": "Embeddings are dense vector representations of text. They enable semantic similarity search in NLP applications."
    }
]

# ----------------------
# Embeddings and Vector DB
# ----------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_vectorstore(documents, persist=True):
    texts = [doc["content"] for doc in documents]
    metadatas = [{"title": doc["title"], "id": doc["id"]} for doc in documents]
    embeddings = get_embedding_model()
    db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    if persist:
        db.save_local(INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)
    return db

def load_or_build_vectorstore():
    if os.path.exists(f"{INDEX_PATH}/index.faiss"):
        embeddings = get_embedding_model()
        return FAISS.load_local(INDEX_PATH, embeddings)
    else:
        return build_vectorstore(DOCUMENTS)

# ----------------------
# LangChain QA Chain
# ----------------------
def build_qa_chain(vectorstore):
    prompt_template = """Use the context below to answer the question.
If the answer is not found in the context, respond with "I don't know."

Context:
{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# ----------------------
# Streamlit App
# ----------------------
def main():
    st.set_page_config(page_title="RAG QA", layout="centered")
    st.title("🔍 Retrieval-Augmented Q&A with OpenAI + FAISS")

    if not OPENAI_API_KEY:
        st.error("⚠️ Please set your OPENAI_API_KEY as an environment variable.")
        return

    # Load or create vectorstore
    vectorstore = load_or_build_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    query = st.text_input("💬 Ask your question:", placeholder="e.g. What is vector search?")
    if query:
        with st.spinner("Generating answer..."):
            response = qa_chain.run(query)
        st.success(response)

        with st.expander("📄 Retrieved Documents"):
            docs = vectorstore.similarity_search(query, k=3)
            for doc in docs:
                st.markdown(f"**{doc.metadata['title']}**\n\n{doc.page_content[:300]}...\n")

if __name__ == "__main__":
    main()




### or we can #####

import os
import faiss
import streamlit as st
import pickle
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, HuggingFaceHub
from PyPDF2 import PdfReader

# ---------------- Configuration ---------------- #
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "vectorstore"
DOCS_PATH = "docs.pkl"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set in env or .env

# ---------------- Document Upload ---------------- #
def load_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = uploaded_file.read().decode("utf-8")
    return text

# ---------------- Embedding Model ---------------- #
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ---------------- Vector Store ---------------- #
def build_vectorstore(documents: List[str], metadatas: List[dict], persist=True):
    embeddings = get_embedding_model()
    db = FAISS.from_texts(documents, embedding=embeddings, metadatas=metadatas)
    if persist:
        db.save_local(INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(list(zip(documents, metadatas)), f)
    return db

def load_vectorstore():
    if os.path.exists(f"{INDEX_PATH}/index.faiss"):
        return FAISS.load_local(INDEX_PATH, get_embedding_model())
    return None

# ---------------- LLM Selection ---------------- #
def get_llm(source="OpenAI", temperature=0.2):
    if source == "OpenAI":
        return OpenAI(temperature=temperature, openai_api_key=OPENAI_API_KEY)
    else:
        return HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": temperature})

# ---------------- RAG Chain ---------------- #
def build_rag_chain(llm, vectorstore, top_k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    prompt_template = """
Use the following context to answer the question.
If the answer isn't found, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

# ---------------- Streamlit App ---------------- #
def main():
    st.set_page_config("🧠 RAG Q&A", layout="wide")
    st.title("🔍 Retrieval-Augmented Q&A App")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        llm_source = st.selectbox("LLM Source", ["OpenAI", "HuggingFace"])
        top_k = st.slider("Top-k Documents", 1, 5, 3)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2)
        st.markdown("---")
        uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt"], accept_multiple_files=True)

    # Load or create vectorstore
    vectorstore = load_vectorstore()
    if uploaded_files:
        texts = []
        metas = []
        for file in uploaded_files:
            text = load_uploaded_file(file)
            texts.append(text)
            metas.append({"filename": file.name})
        vectorstore = build_vectorstore(texts, metas)
        st.success("📚 Documents processed and added to FAISS index.")

    if not vectorstore:
        st.warning("Please upload documents to start.")
        return

    llm = get_llm(source=llm_source, temperature=temperature)
    qa_chain = build_rag_chain(llm, vectorstore, top_k)

    # Main interface
    query = st.text_input("Ask your question here:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
        st.success(answer)

        with st.expander("📄 Retrieved Contexts"):
            docs = vectorstore.similarity_search(query, k=top_k)
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Doc {i} — {doc.metadata.get('filename', '')}**")
                st.markdown(doc.page_content[:300] + "…")

if __name__ == "__main__":
    main()
