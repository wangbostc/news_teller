# This script is used to populate the database with the scraped news content.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_transformers import Html2TextTransformer
from news_teller.config import NEWS_URLS, DB_PATH, TOP_K, OPENAI_API_KEY
from news_teller.news_loader import (scrape_news_html_content, 
                                     process_raw_html_doc,
                                     )

# scrape from the web
news_html_content = scrape_news_html_content(NEWS_URLS)

# process the html doc
html2text_transfomer = Html2TextTransformer()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = process_raw_html_doc(raw_html_doc=news_html_content,
                     html2text_transfomer=html2text_transfomer ,
                     text_splitter=text_splitter,
                     )

# store the chunks in the vectorstore
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
vector_store = Chroma.from_documents(documents=chunks, 
                                     embedding=embeddings_model, 
                                     collection_name="news", 
                                     persist_directory=DB_PATH)
    

