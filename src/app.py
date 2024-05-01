from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

from fastapi import FastAPI
import re
import uvicorn

from templates import TEMPLATE_RAG, TEMPLATE_CHAT
from schemas import Request
from utils import format_docs, add_data, get_response_dict


embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_rag = PromptTemplate.from_template(TEMPLATE_RAG)
prompt_chat = PromptTemplate.from_template(TEMPLATE_CHAT)

llm = Ollama(model="EEVE-Korean-10.8B:v2")


chain_rag = (
    {"context": retriever | format_docs, "prompt": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

chain_chat = {"prompt": RunnablePassthrough()} | prompt_chat | llm | StrOutputParser()

app = FastAPI()
ADD_DATA_PREFIX = "/데이터입력"


@app.post("/chat")
async def chat(request: Request):
    req_text = request.userRequest.utterance

    response_text = chain_chat.invoke(req_text)

    response_text = response_text.strip()

    res = get_response_dict(response_text)

    return res


@app.post("/")
async def root(request: Request):
    req_text = request.userRequest.utterance

    if req_text.startswith(ADD_DATA_PREFIX):
        response_text = add_data(retriever, req_text, prefix=ADD_DATA_PREFIX)
    else:
        response_text = chain_rag.invoke(req_text)

    res = get_response_dict(response_text.strip())

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
