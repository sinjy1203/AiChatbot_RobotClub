import os

import uvicorn
from fastapi import FastAPI

from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain_community.llms import VLLMOpenAI

from templates import (
    TEMPLATE_RETRIEVAL_GRADE,
    TEMPLATE_CONTEXTUAL_COMPRESSION,
    TEMPLATE_GENERATE,
)
from schemas import Request, Response
from nodes import (
    RetrieveNode,
    GradeRetrievalNode,
    CompressNode,
    GenerateNode,
    null_generate_node,
)
from edges import decide_to_generate
from states import GraphState

os.environ["LANGCHAIN_TRACING_V2"] = (
    "false" if os.environ["LANGCHAIN_TRACING_V2"] == "f" else "true"
)

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
HUGGINGFACE_CACHE_FOLDER = os.environ["HUGGINGFACE_CACHE_FOLDER"]
LLM_API_BASE = os.environ["LLM_API_BASE"]
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
MODEL_NAME = os.environ["MODEL_NAME"]

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=LLM_API_BASE,
    model_name=MODEL_NAME,
    model_kwargs={"stop": ["."]},
)

retrieve = RetrieveNode(
    embedding_model=EMBEDDING_MODEL,
    cache_folder=HUGGINGFACE_CACHE_FOLDER,
    chromadb_host=CHROMADB_HOST,
    top_k=3,
)

grade_retrieval = GradeRetrievalNode(llm=llm, template=TEMPLATE_RETRIEVAL_GRADE)

compress = CompressNode(llm=llm, template=TEMPLATE_CONTEXTUAL_COMPRESSION)

generate = GenerateNode(llm=llm, template=TEMPLATE_GENERATE)

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve.node)
workflow.add_node("grade_retrieval", grade_retrieval.node)
workflow.add_node("compress", compress.node)
workflow.add_node("generate", generate.node)
workflow.add_node("null_generate", null_generate_node)


workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_retrieval")
workflow.add_conditional_edges(
    "grade_retrieval",
    decide_to_generate,
    {
        "generate": "compress",
        "null_generate": "null_generate",
    },
)
workflow.add_edge("compress", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("null_generate", END)

graph = workflow.compile()


app = FastAPI()


@app.post("/contexts")
async def contexts(request: Request) -> Response:
    retrieve.retriever.add_documents([Document(page_content=request.text)])
    return Response(text="데이터 입력이 완료하였습니다.")


@app.post("/rag")
async def rag(request: Request):
    response_text = graph.invoke({"question": request.text})["generation"]
    return Response(text=response_text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1200)
