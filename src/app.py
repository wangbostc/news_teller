from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from scraper import scrape_news
from retriever import NewsRetriever
from model import construct_chain, llm
from config import NEWS_URLS
from langchain.chains.base import Chain


def initialize_retriever():
    try:
        news_content = scrape_news(NEWS_URLS)
        retriever = NewsRetriever(news_content)
        retriever.store_documents(presist=True)
        retri = retriever.get_retriever(from_presist=True)
        return construct_chain(retri, llm)
    except RuntimeError as e:
        print(f"Error during initialization: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    chain = initialize_retriever()
    yield {"chain": chain}


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    return {"This is a news chatbot"}


@app.post("/olymics_news_chatbot")
async def query(request: Request, request_body: QueryRequest):
    """
    Process a user query and return the latest news based on the query.
    """
    
    user_query = request_body.query
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required")
    try:
        chain = request.state.chain
        response = chain.invoke(user_query)
        return {"Latest news": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
