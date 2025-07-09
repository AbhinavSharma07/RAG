from fastapi import APIRouter, File, UploadFile
from PyPDF2 import PdfReader
from core.embeddings import embed_texts
from core.retrieval import store_embeddings

router = APIRouter()

@router.post("/")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    reader = PdfReader(file.file)
    pages = [page.extract_text() for page in reader.pages]
    
    embeddings = embed_texts(pages)
    store_embeddings(pages, embeddings)
    
    return {"msg": "Uploaded and embedded"}
