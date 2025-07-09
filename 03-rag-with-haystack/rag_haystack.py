from haystack.nodes import FARMReader, TransformersRetriever, PreProcessor
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline

# 1. Initialize Document Store
document_store = InMemoryDocumentStore()

# 2. Sample documents
docs = [
    {"content": "Paris is the capital of France."},
    {"content": "The Eiffel Tower is located in Paris."},
    {"content": "France is known for wine and cheese."}
]

# 3. Write documents
document_store.write_documents(docs)

# 4. Initialize Retriever and Reader
retriever = TransformersRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
document_store.update_embeddings(retriever)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# 5. Create Pipeline
pipe = ExtractiveQAPipeline(reader, retriever)

# 6. Ask Question
prediction = pipe.run(query="Where is the Eiffel Tower?", params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 1}})

print("Answer:", prediction["answers"][0].answer)
