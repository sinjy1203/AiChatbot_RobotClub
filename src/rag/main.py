import os
import uvicorn
from fastapi import FastAPI

from schemas import Request
from utils import get_response_dict, del_prefix
import graph

os.environ["LANGCHAIN_TRACING_V2"] = (
    "false" if os.environ["LANGCHAIN_TRACING_V2"] == "f" else "true"
)

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
HUGGINGFACE_CACHE_FOLDER = os.environ["HUGGINGFACE_CACHE_FOLDER"]
LLM_API_BASE = os.environ["LLM_API_BASE"]
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
MODEL_NAME = os.environ["MODEL_NAME"]

ADD_DATA_PREFIX = "/데이터입력"

rag_graph = graph.generate(
    embedding_model=EMBEDDING_MODEL,
    huggingface_cache_folder=HUGGINGFACE_CACHE_FOLDER,
    llm_api_base=LLM_API_BASE,
    chromadb_host=CHROMADB_HOST,
    llm_model=MODEL_NAME,
    top_k=3,
)


app = FastAPI()


@app.post("/")
async def root(request: Request):
    req_text = request.userRequest.utterance

    if req_text.startswith(ADD_DATA_PREFIX):
        req_text = del_prefix(req_text, ADD_DATA_PREFIX)
        request_state = {"new_context": req_text}
    else:
        request_state = {"question": req_text}

    response_state = rag_graph.invoke(request_state)
    response_text = response_state["generation"]

    res = get_response_dict(response_text.strip())

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
