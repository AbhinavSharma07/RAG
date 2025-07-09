from PyPDF2 import PdfReader
from typing import List
import re

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    # Very simple tokenizer: split by sentence (can replace with tokenizer)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = []

    tokens = 0
    for sentence in sentences:
        sentence_tokens = len(sentence.split())  # Replace with tiktoken if needed
        if tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [sentence]
            tokens = sentence_tokens
        else:
            chunk.append(sentence)
            tokens += sentence_tokens

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks
