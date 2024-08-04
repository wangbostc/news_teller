from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from news_teller.model import construct_chain
from news_teller.config import DB_PATH, TOP_K


def make_app():
    # setting up required chain
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm_model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini-2024-07-18")

    retriever = Chroma(
        persist_directory=DB_PATH,
        collection_name="news",
        embedding_function=embedding_function,
    ).as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    chain = construct_chain(retriever, llm_model)
    app = FastAPI()

    class AppQuery(BaseModel):
        query: str

    @app.get("/health_check")
    async def get_health():
        return "ok"

    @app.post("/olymics_news_chatbot")
    async def handle_query(query: AppQuery):
        """
        Process a user query and return the latest news based on the query.
        """

        user_query = query.query
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
        try:
            response = chain.invoke(user_query)
            return {"Latest news": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = make_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
