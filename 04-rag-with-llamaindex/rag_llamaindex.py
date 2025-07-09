from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SimpleNodeParser
import faiss

# Load documents
docs = ["Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "France is famous for wine and cheese."]

# Save as text
with open("llm_docs.txt", "w") as f:
    for line in docs:
        f.write(f"{line}\n")

# Read from file
reader = SimpleDirectoryReader(input_files=["llm_docs.txt"])
documents = reader.load_data()

# Embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Parse into nodes
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Create FAISS Index
dimension = embed_model.get_query_embedding("test").shape[0]
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)
index = VectorStoreIndex(nodes, embed_model=embed_model, vector_store=vector_store)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Where is the Eiffel Tower?")
print("Answer:", response)
