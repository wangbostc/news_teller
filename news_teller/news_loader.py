from typing import List, Union

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document, BaseDocumentTransformer
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document

from news_teller.config import DB_PATH


def scrape_news_html_content(urls: Union[List[str], str]) -> str:
    try:
        loader = AsyncHtmlLoader(urls)
        html_content = loader.load()
        return html_content
    except Exception as e:
        raise RuntimeError(f"Error scraping news: {e}")


def process_raw_html_doc(
    raw_html_doc: Document,
    html2text_transfomer: BaseDocumentTransformer,
    text_splitter: TextSplitter,
) -> List[Document]:
    """
    Process the document by splitting it into chunks, embedding it, and storing it in a vectorstore.
    """
    structured_text_docutm = html2text_transfomer.transform_documents(raw_html_doc)
    chunks = text_splitter.split_documents(structured_text_docutm)
    return chunks
