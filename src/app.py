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

from templates import TEMPLATE
from utils import format_docs, add_data, get_response_dict


embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = PromptTemplate.from_template(TEMPLATE)

llm = Ollama(model="EEVE-Korean-10.8B:v2")


rag_chain = (
    {"context": retriever | format_docs, "prompt": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()
ADD_DATA_PREFIX = "/데이터입력"


@app.post("/")
async def root(request: dict):
    req = dict(request)
    req_text = req["userRequest"]["utterance"]
    if req_text.startswith(ADD_DATA_PREFIX):
        response_text = add_data(retriever, req_text, prefix=ADD_DATA_PREFIX)
    else:
        response_text = rag_chain.invoke(req_text)

    response_text = response_text.strip()

    res = get_response_dict(response_text)

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
