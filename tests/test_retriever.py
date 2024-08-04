import pytest
from retriever import NewsRetriever
from langchain_core.documents import Document

@pytest.fixture
def document():
    return [Document(
        page_content='This is a test doc.' * 100, 
        metadata={"source": "http://test.com"}),
            Document(
        page_content='vectorstore is very fast.', 
        metadata={"source": "http://test.com"}),
            ]

@pytest.mark.parametrize('chunk_size, chunk_overlap, expected_chunks', [
    (100, 0, 21), # 20 + 1
    (100, 20, 26), # 25 + 1
    (1000, 0, 3), # 2 + 1
])
def test_chunk_document(document, chunk_size, chunk_overlap, expected_chunks):
    
    retriever = NewsRetriever(document=document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    assert len(retriever.chunks) ==  expected_chunks 


@pytest.mark.parametrize('chunk_size, chunk_overlap, top_k, expected_num_of_retrievals', [
    (100, 0, 3, 3),
    (100, 0, 5, 5),
])
def test_retrieve_top_k(document, chunk_size, chunk_overlap, top_k, expected_num_of_retrievals):
    retriever =  NewsRetriever(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
    retriever.store_documents(presist=False)
    result=retriever.get_retriever(from_presist=False).invoke("vectorstore is good.")
    assert len(result) == expected_num_of_retrievals
    assert result[0].page_content == 'vectorstore is very fast.'