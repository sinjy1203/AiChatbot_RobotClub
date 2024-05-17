import os

import uvicorn
from fastapi import FastAPI
import chromadb
from chromadb.config import Settings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.llms import VLLMOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils import format_docs
from templates import TEMPLATE_RAG, TEMPLATE_CHAT
from schemas import Request, Response

os.environ["LANGCHAIN_TRACING_V2"] = (
    "false" if os.environ["LANGCHAIN_TRACING_V2"] == "f" else "true"
)

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
LLM_API_BASE = os.environ["LLM_API_BASE"]
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
MODEL_NAME = os.environ["MODEL_NAME"]

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask", cache_folder=EMBEDDING_MODEL
)

client = chromadb.HttpClient(host=CHROMADB_HOST, settings=Settings(allow_reset=True))
vectorstore = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_rag = PromptTemplate.from_template(TEMPLATE_RAG)
prompt_chat = PromptTemplate.from_template(TEMPLATE_CHAT)

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=LLM_API_BASE,
    model_name=MODEL_NAME,
    model_kwargs={"stop": ["."]},
)

chain_rag = (
    {"context": retriever | format_docs, "prompt": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

chain_chat = {"prompt": RunnablePassthrough()} | prompt_chat | llm | StrOutputParser()

app = FastAPI()


@app.post("/chat")
async def chat(request: Request) -> Response:
    response_text = chain_chat.invoke(request.text).strip()
    return Response(text=response_text)


@app.post("/contexts")
async def contexts(request: Request) -> Response:
    retriever.add_documents([Document(page_content=request.text)])
    return Response(text="데이터 입력이 완료하였습니다.")


@app.post("/rag")
async def rag(request: Request):
    response_text = chain_rag.invoke(request.text).strip()
    return Response(text=response_text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1200)
