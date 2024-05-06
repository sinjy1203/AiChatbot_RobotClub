from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.llms import VLLMOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from fastapi import FastAPI
import re
import uvicorn

from utils import format_docs
from templates import TEMPLATE_RAG, TEMPLATE_CHAT
from schemas import Request, Response


embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_rag = PromptTemplate.from_template(TEMPLATE_RAG)
prompt_chat = PromptTemplate.from_template(TEMPLATE_CHAT)

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    model_name="EEVE-Korean-Instruct-10.8B-v1.0-quantized",
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


@app.post("/")
async def root(request: Request):
    response_text = chain_rag.invoke(request.text).strip()
    return Response(text=response_text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1200)
