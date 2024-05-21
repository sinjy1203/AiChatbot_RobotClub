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


class RetrieveNode:
    def __init__(
        self,
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
        self.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )

        self.chain = self.retriever | RunnableLambda(format_docs)

    def node(self, state):
        question = state["question"]
        contexts = self.chain.invoke(question)
        state["contexts"] = contexts
        return state


class GradeRetrievalNode:
    def __init__(self, llm, template=TEMPLATE_RETRIEVAL_GRADE):
        prompt = PromptTemplate(
            template=template, input_variables=["question", "contexts"]
        )
        self.chain = prompt | llm | BinaryOutputParser()

    def node(self, state):
        question = state["question"]
        contexts = state["contexts"]

        res = self.chain.invoke({"question": question, "contexts": contexts})
        if res == "예":
            state["retrieved"] = True
        else:
            state["retrieved"] = False
        return state


class CompressNode:
    def __init__(self, llm, template=TEMPLATE_CONTEXTUAL_COMPRESSION):
        prompt = PromptTemplate(
            template=template, input_variables=["question", "contexts"]
        )
        self.chain = prompt | llm | StrOutputParser()

    def node(self, state):
        question = state["question"]
        contexts = state["contexts"]
        summarized_context = self.chain.invoke(
            {"question": question, "contexts": contexts}
        )
        state["summarized_context"] = summarized_context
        return state


class GenerateNode:
    def __init__(self, llm, template=TEMPLATE_GENERATE):
        prompt = PromptTemplate(
            template=template, input_variables=["question", "context"]
        )
        self.chain = prompt | llm | StrOutputParser()

    def node(self, state):
        question = state["question"]
        summarized_context = state["summarized_context"]

        generation = self.chain.invoke(
            {"question": question, "contexts": summarized_context}
        )
        state["generation"] = generation
        return state


def null_generate_node(state):
    state["generation"] = "정보가 부족합니다."
    return state
