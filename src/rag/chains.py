from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings

from utils import format_docs, BinaryOutputParser
from templates import (
    TEMPLATE_RETRIEVAL_GRADE,
    TEMPLATE_CONTEXTUAL_COMPRESSION,
    TEMPLATE_GENERATE,
)


def get_retriever_chain(
    embedding_model="jhgan/ko-sroberta-multitask",
    cache_folder="/embedding_model",
    chromadb_host="vectorstore-service.default",
    top_k=3,
):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model, cache_folder=cache_folder
    )

    client = chromadb.HttpClient(
        host=chromadb_host, settings=Settings(allow_reset=True)
    )
    vectorstore = Chroma(
        client=client,
        collection_name="my_collection",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

    return retriever | RunnableLambda(format_docs)


def get_retrieval_grader_chain(llm, template=TEMPLATE_RETRIEVAL_GRADE):
    prompt = PromptTemplate.from_template(
        template, input_variables=["question", "contexts"]
    )
    return prompt | llm | BinaryOutputParser()


def get_contextual_compression_chain(llm, template=TEMPLATE_CONTEXTUAL_COMPRESSION):
    prompt = PromptTemplate.from_template(
        template, input_variables=["question", "contexts"]
    )
    return prompt | llm | StrOutputParser()


def get_generate_chain(llm, template=TEMPLATE_GENERATE):
    prompt = PromptTemplate.from_template(
        template, input_variables=["question", "context"]
    )
    return prompt | llm | StrOutputParser()
