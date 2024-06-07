import os
import uvicorn
from fastapi import FastAPI

from schemas import Request, Document, DocumentIds
from utils import get_response_dict, get_vectorstore
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

vectorstore = get_vectorstore(
    chromadb_host=CHROMADB_HOST, collection_name="my_collection"
)

rag_graph = graph.generate(
    embedding_model=EMBEDDING_MODEL,
    huggingface_cache_folder=HUGGINGFACE_CACHE_FOLDER,
    llm_api_base=LLM_API_BASE,
    chromadb_host=CHROMADB_HOST,
    llm_model=MODEL_NAME,
    top_k=3,
)


app = FastAPI()


@app.post("/input")
async def input(request: Request):
    req_text = request.userRequest.utterance
    request_state = {"new_context": req_text}

    response_state = rag_graph.invoke(request_state)
    response_text = response_state["generation"]
    res = get_response_dict(response_text.strip())

    return res


@app.post("/output")
async def output(request: Request):
    req_text = request.userRequest.utterance
    request_state = {"question": req_text}

    response_state = rag_graph.invoke(request_state)
    response_text = response_state["generation"]

    res = get_response_dict(response_text.strip())

    return res


@app.get("/documents")
async def documents() -> list[Document]:
    docs = vectorstore.get()
    res = [
        Document(id=docs["ids"][i], document=docs["documents"][i])
        for i in range(len(docs["ids"]))
    ]
    return res


@app.post("/delete")
async def delete(document_ids: DocumentIds) -> str:
    vectorstore.delete(ids=document_ids.ids)
    return "Deleted"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
