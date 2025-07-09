import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Simple RAG setup
docs = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "France is known for cheese and wine."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def rag_answer(query):
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, I = index.search(query_embedding, 2)
    context = " ".join([docs[i] for i in I[0]])
    return f"Context: {context}\n\nAnswer (basic): {context.split('.')[0]}."

# Gradio interface
iface = gr.Interface(fn=rag_answer,
                     inputs=gr.Textbox(label="Ask a question"),
                     outputs="text",
                     title="RAG Demo (Basic)",
                     description="Retrieval-Augmented Generation demo using FAISS and Sentence Transformers")

if __name__ == "__main__":
    iface.launch()
