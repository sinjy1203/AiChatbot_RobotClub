from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

from fastapi import FastAPI, HTTPException
import re
import json
import uvicorn
from pydantic import BaseModel

import ollama

embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

TEMPLATE = """
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {prompt}
"""
prompt = PromptTemplate.from_template(TEMPLATE)

llm = Ollama(model="EEVE-Korean-10.8B:v2")


def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


rag_chain = (
    {"context": retriever | format_docs, "prompt": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()


@app.post("/")
async def root(request: dict):
    req = dict(request)
    req_text = req["userRequest"]["utterance"]
    print(req_text)
    if req_text.startswith("/데이터입력"):
        pattern = "^" + re.escape("/데이터입력")
        req_text = re.sub(pattern, "", req_text).strip()

        retriever.add_documents([Document(page_content=req_text)])
        response_text = "데이터 입력이 완료하였습니다."
    else:
        response_text = rag_chain.invoke(req_text)

    response_text = response_text.strip()
    print(response_text)
    res = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": response_text}}]},
    }

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
