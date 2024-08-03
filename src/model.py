from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
from retriever import NewsRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from utils import format_docs

llm = ChatOpenAI(
    temperature=0.0, api_key=OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18"
)


def construct_chain(retriever: NewsRetriever, llm: BaseChatModel) -> str:
    """construct the rag chain for querying"""

    template = """ Assume you are a journalist reporting the latest news about the Olympics. 
    You have access to the following information:
    Context: {context}
    ----------------------------------------------
    Please based on the query below, provide a detailed response. If you can not find the information in the above context, say "I don't have that information".
    Query: {query}
    
    Answer:
    """

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "query": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
