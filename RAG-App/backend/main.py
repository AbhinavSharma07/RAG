from fastapi import FastAPI
from api.upload import router as upload_router
from api.query import router as query_router

app = FastAPI()

app.include_router(upload_router, prefix="/upload", tags=["upload"])
app.include_router(query_router, prefix="/query", tags=["query"])

@app.get("/")
def read_root():
    return {"msg": "RAG API running"}
