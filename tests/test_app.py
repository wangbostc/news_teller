import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app, construct_chain

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def mock_chain():
    # Create a mock chain object
    mock = MagicMock()
    # Configure the mock to return a specific response
    mock.invoke.return_value = "Mocked news content"
    # Override the global `chain` with the mock
    global chain
    chain = mock
    yield
    # Cleanup if needed (e.g., reset chain to its original state)
    del chain

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ["This is a news chatbot"]

@pytest.mark.skip(reason="Not implemented")
@patch("app.construct_chain")
def test_query_endpoint_valid_request(mock_chain_constructor):
    # Mock the chain object
    mock_chain = mock_chain_constructor.return_value
    mock_chain.invoke.return_value = "This is the latest news."

    response = client.post("/olymics_news_chatbot", json={"query": "latest news"})
    
    assert response.status_code == 200
    assert response.json() == {"Latest news": "This is the latest news."}

@pytest.mark.skip(reason="Not implemented")
def test_query_endpoint_no_retriever():
    with pytest.raises(RuntimeError):
        response = client.post("/olymics_news_chatbot", json={"query": "latest news"})
        assert response.status_code == 500
        assert response.json() == {"detail": "News content not available"}

@pytest.mark.skip(reason="Not implemented")
def test_query_endpoint_invalid_request():
    response = client.post("/olymics_news_chatbot", json={"query": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Query is required"}
