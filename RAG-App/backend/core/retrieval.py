import faiss
import numpy as np

index = None
stored_texts = []

def store_embeddings(texts, embeddings):
    global index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    stored_texts.extend(texts)

def retrieve_similar(query):
    from core.embeddings import embed_texts
    embedding = embed_texts([query])[0]
    D, I = index.search(np.array([embedding]).astype("float32"), 5)
    return [stored_texts[i] for i in I[0]]
