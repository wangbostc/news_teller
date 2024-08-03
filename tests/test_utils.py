import pytest
from utils import format_docs
from langchain_core.documents import Document

@pytest.fixture
def documents():
    return [
        Document(page_content="doc1", 
                 metadata={"source": "http://test.com"}),
        Document(page_content="doc2", 
                 metadata={"source": "http://test.com"}),
    ]   
def test_format_docs(documents):
    result = format_docs(documents)
    assert result == "doc1\n\ndoc2"
