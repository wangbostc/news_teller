from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from news_teller.utils import format_docs


def construct_chain(retriever: VectorStoreRetriever, llm: BaseChatModel) -> str:
    """construct the rag chain for querying"""

    template = """ Assume you are a journalist reporting the latest news about the Olympics. 
    You have access to the following information:
    Context: {context}
    ----------------------------------------------
    Please based on the query below, provide a detailed response. 
    If you can not find the information in the above context, say "I don't have that information".
    Query: {query}
    
    Answer:
    """

    prompt = PromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "query": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
