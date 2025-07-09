import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def embed_texts(texts):
    embeddings = []
    for text in texts:
        res = client.embeddings.create(input=[text], model="text-embedding-3-small")
        embeddings.append(res.data[0].embedding)
    return embeddings
