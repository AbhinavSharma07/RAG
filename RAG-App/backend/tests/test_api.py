import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_upload_missing_file():
    response = client.post("/upload/")
    assert response.status_code == 422

def test_query_empty():
    response = client.post("/query/", json={"question": ""})
    assert response.status_code in [400, 422]
