from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scraper import scrape_news
from retriever import NewsRetriever
from model import construct_chain, llm
from config import NEWS_URLS


def initialize_retriever():
    global chain
    try:
        news_content = scrape_news(NEWS_URLS)
        retriever = NewsRetriever(news_content).get_retriever()
        chain = construct_chain(retriever, llm)
    except RuntimeError as e:
        print(f"Error during initialization: {e}")
        retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_retriever()
    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    return {"This is a news chatbot"}


@app.post("/olymics_news_chatbot")
async def query(request: QueryRequest):
    if chain is None:
        raise HTTPException(status_code=500, detail="News content not available")

    try:
        user_query = request.query
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")

        response = chain.invoke(user_query)
        return {"Latest news": response}
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
