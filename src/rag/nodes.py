from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
import uuid

from utils import format_docs, BinaryOutputParser
from templates import (
    TEMPLATE_RETRIEVAL_GRADE,
    TEMPLATE_CONTEXTUAL_COMPRESSION,
    TEMPLATE_GENERATE,
    TEMPLATE_GENERATE_QUESTION,
)


class GenerateQuestionNode:
    def __init__(self, llm, template=TEMPLATE_GENERATE_QUESTION):
        prompt = PromptTemplate(template=template, input_variables=["new_context"])
        self.chain = prompt | llm | StrOutputParser()

    def node(self, state):
        new_document = state["new_context"]
        question = self.chain.invoke(new_document)
        state["question"] = question
        return state


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
        self.vectorstore = Chroma(
            client=client,
            collection_name="my_collection",
            embedding_function=embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )

        self.chain = self.retriever

    def node(self, state):
        question = state["question"]
        docs = self.chain.invoke(question)
        contexts = format_docs(docs)
        state["docs"] = docs
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

        if len(state["docs"]) == 0:
            state["retrieved"] = False
            return state

        res = self.chain.invoke({"question": question, "contexts": contexts})
        if res == "예":
            state["retrieved"] = True
        else:
            state["retrieved"] = False
        return state


class UpdateContextNode:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def node(self, state):
        old_doc = state["docs"][0]
        old_doc.page_content = state["new_context"]

        self.vectorstore.update_document(
            old_doc.metadata["id"],
            old_doc,
        )
        state["generation"] = "데이터 업데이트가 완료되었습니다."
        return state


class AddContextNode:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def node(self, state):
        new_doc = Document(
            page_content=state["new_context"], metadata={"id": uuid.uuid1().hex}
        )
        self.vectorstore.add_documents([new_doc], ids=[new_doc.metadata["id"]])
        state["generation"] = "데이터가 추가되었습니다."
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
