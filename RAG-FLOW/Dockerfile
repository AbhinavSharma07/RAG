FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
docker build -t rag-app .
docker run -p 8501:8501 rag-app
