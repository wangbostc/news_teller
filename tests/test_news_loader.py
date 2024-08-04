from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from news_teller.news_loader import scrape_news_html_content, process_raw_html_doc

@pytest.fixture
def html_content_document():
    return [Document(
        page_content='<html>test html content</html>',
        metadata={"source": "http://test.com"}
    )]

@pytest.fixture
def text_document():
    return [Document(
        page_content='test html content\n\n',
        metadata={"source": "http://test.com"}
    )]
    
@pytest.fixture
def text_documents():
    return [Document(
        page_content='This is a test doc.' * 100, 
        metadata={"source": "http://test.com"}),
            Document(
        page_content='vectorstore is very fast.', 
        metadata={"source": "http://test.com"}),
            ]
# Test scrape_news function
@patch('langchain_community.document_loaders.AsyncHtmlLoader.load')
def test_scrape_news_html_content( mock_async_html_loader, html_content_document, text_document):
    # Mock the behavior of AsyncHtmlLoader
    mock_async_html_loader.return_value = html_content_document
    
    # Test with a list of URLs
    urls = ['http://test.com']
    result = scrape_news_html_content(urls)
    assert result == html_content_document
    
    # Test with a single URL
    url = 'http://test.com'
    result = scrape_news_html_content(url)
    assert result == html_content_document
    
    # Test exception handling
    mock_async_html_loader.side_effect = Exception('Loader error')
    with pytest.raises(RuntimeError, match='Error scraping news: Loader error'):
        scrape_news_html_content(url)
        
        
@pytest.mark.parametrize('chunk_size, chunk_overlap, expected_chunks', [
    (100, 0, 26),
    (500, 50, 6), 
    (1000, 100, 3), 
])
def test_process_raw_html_doc(text_documents, 
                              chunk_size, 
                              chunk_overlap, 
                              expected_chunks):
    html2text_transfomer = Html2TextTransformer()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = process_raw_html_doc(text_documents, 
                                  html2text_transfomer, 
                                  text_splitter)
    
    assert len(chunks) ==  expected_chunks 
