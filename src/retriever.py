from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.documents import Document
from config import OPENAI_API_KEY, DB_PATH


class NewsRetriever:
    def __init__(
        self,
        document: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        top_k: int = 3,
        embedding_model: Optional[Embeddings] = None,
        vectorstore_type: Optional[VectorStore] = None,
    ):
        """
        Initialize the NewsRetriever by chunking the document, embedding it, and loading it into a vectorstore.

        Args:
            document (str): HTML content.
            chunk_size (int, optional): Size of each chunk. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 0.
            top_k (int, optional): Number of top results to retrieve. Defaults to 3.
            embedding_model (Embeddings, optional): Embedding model to use. Defaults to OpenAIEmbeddings.
            splitter_class (class, optional): Class to use for splitting text. Defaults to RecursiveCharacterTextSplitter.
            vectorstore_class (class, optional): Class to use for vectorstore. Defaults to Chroma.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.document = document
        self.embedding_model = embedding_model or OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
        )
        self.chunks = self.chunk_document()
        self.vectorstore_type = vectorstore_type or Chroma
        self.vectorstore = None

    def chunk_document(self) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(self.document)

    def store_documents(self, presist: bool = True) -> VectorStore:
        if presist:
            self.vectorstore = self.vectorstore_type.from_documents(
                documents=self.chunks,
                embedding=self.embedding_model,
                persist_directory=DB_PATH,
            )
        else:
            self.vectorstore = self.vectorstore_type.from_documents(
                documents=self.chunks, embedding=self.embedding_model
            )
        return self.vectorstore

    def get_retriever(self, from_presist: bool = False) -> VectorStoreRetriever:
        if from_presist:
            self.vectorstore = Chroma(
                persist_directory=DB_PATH, embedding_function=self.embedding_model
            )
        return self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )
