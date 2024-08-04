import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app

with TestClient(app) as client:

    def test_root_endpoint():
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == ["This is a news chatbot"]

    def test_query_endpoint_success():
        # Test with a valid query
        query_data = {"query": "latest olympics news"}
        response = client.post("/olymics_news_chatbot", json=query_data)
        assert response.status_code == 200
        assert "Latest news" in response.json()

    def test_query_endpoint_invalid_input():
        # test no input
        response = client.post("/olymics_news_chatbot", json={"query": ""})
        assert response.status_code == 400
        assert response.json() == {"detail": "Query is required"}


def test_query_endpoint_no_chain():
    client_no_chain = TestClient(app)
    response = client_no_chain.post(
        "/olymics_news_chatbot", json={"query": "latest news"}
    )
    assert response.status_code == 500
