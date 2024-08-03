import pytest
from unittest.mock import patch
from scraper import scrape_news, extract_and_structure_text 
from langchain_core.documents import Document

@pytest.fixture
def html_content_document():
    return [Document(
        page_content='<html>test html content</html>',
        metadata={"source": "http://test.com"}
    )]

@pytest.fixture
def text_dcoument():
    return [Document(
        page_content='test html content\n\n',
        metadata={"source": "http://test.com"}
    )]
    

# Test scrape_news function
@patch('src.scraper.AsyncHtmlLoader.load')
def test_scrape_news( mock_async_html_loader, html_content_document, text_dcoument):
    # Mock the behavior of AsyncHtmlLoader
    mock_async_html_loader.return_value = html_content_document
    
    # Test with a list of URLs
    urls = ['http://test.com']
    result = scrape_news(urls)
    assert result == text_dcoument
    
    # Test with a single URL
    url = 'http://test.com'
    result = scrape_news(url)
    assert result == text_dcoument
    
    # Test exception handling
    mock_async_html_loader.side_effect = Exception('Loader error')
    with pytest.raises(RuntimeError, match='Error scraping news: Loader error'):
        scrape_news(url)

@patch('src.scraper.Html2TextTransformer')
def test_extract_and_structure_text(mock_html2text_transformer, html_content_document):
    # Mock the Html2TextTransformer
    mock_html2text_transformer.transform_documents.return_value = html_content_document
    
    result = extract_and_structure_text(html_content_document)
    assert result[0].page_content == 'test html content\n\n'