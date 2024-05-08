import os
from fastapi import FastAPI
import uvicorn

from client import Client
from schemas import Request
from utils import get_response_dict, del_prefix

RAG_BASE_URL = os.environ["RAG_BASE_URL"]
ADD_DATA_PREFIX = "/데이터입력"

client = Client(base_url=RAG_BASE_URL)
app = FastAPI()


@app.post("/chat")
async def chat(request: Request):
    req_text = request.userRequest.utterance
    response_text = client.chat({"text": req_text})["text"].strip()

    res = get_response_dict(response_text)
    return res


@app.post("/")
async def root(request: Request):
    req_text = request.userRequest.utterance

    if req_text.startswith(ADD_DATA_PREFIX):
        req_text = del_prefix(req_text, ADD_DATA_PREFIX)
        response_text = client.contexts({"text": req_text})
    else:
        response_text = client.rag({"text": req_text})

    res = get_response_dict(response_text["text"].strip())

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
