from fastapi import APIRouter
from pydantic import BaseModel
from core.retrieval import retrieve_similar
from openai import OpenAI

client = OpenAI()

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/")
def ask_question(req: QueryRequest):
    retrieved = retrieve_similar(req.question)
    context = "\n".join(retrieved)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Answer with context:"},
            {"role": "user", "content": f"{context}\n\nQuestion: {req.question}"}
        ]
    )
    return {"answer": response.choices[0].message.content}
