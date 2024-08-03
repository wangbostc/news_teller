from typing import List, Union

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from config import USER_AGENT


def scrape_news(urls: Union[List[str], str]) -> str:
    try:
        loader = AsyncHtmlLoader(urls)
        html_content = loader.load()
        return extract_and_structure_text(html_content)
    except Exception as e:
        raise RuntimeError(f"Error scraping news: {e}")


def extract_and_structure_text(html_content: List[Document]) -> List[Document]:
    html2text = Html2TextTransformer()
    structured_text = html2text.transform_documents(html_content)
    return structured_text
