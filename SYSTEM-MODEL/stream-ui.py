import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="🧠 RAG Chatbot", layout="wide")
st.title("🧠 RAG Chatbot (PDF Upload + GPT-4)")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")
if uploaded_file:
    filepath = os.path.join("data", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded and saved.")

    if st.button("Process & Embed"):
        with st.spinner("🔍 Reading and chunking document..."):
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embed)
            db.save_local("vectorstore")
            st.success("📦 Document embedded and stored in vector DB!")

if os.path.exists("vectorstore"):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore", embed)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    prompt = PromptTemplate(input_variables=["context", "question"], template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
""")

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,
                                     chain_type_kwargs={"prompt": prompt})

    query = st.text_input("💬 Ask your question:")
    if query:
        with st.spinner("🧠 Generating answer..."):
            answer = qa.run(query)
            st.markdown("### 🤖 Answer:")
            st.write(answer)





##install dep:
pip install streamlit langchain openai faiss-cpu sentence-transformers PyMuPDF python-dotenv

##env-sample:
OPENAI_API_KEY=your_openai_key_here

##Run the app:
streamlit run app.py
