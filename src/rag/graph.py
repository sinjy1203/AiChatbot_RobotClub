from langchain_community.llms import VLLMOpenAI
from langgraph.graph import END, StateGraph
from states import GraphState

from nodes import *
from edges import *
from states import GraphState
from templates import *


def generate(
    embedding_model,
    huggingface_cache_folder,
    llm_api_base,
    chromadb_host,
    llm_model,
    top_k=3,
):
    llm_grade_retrieval = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=llm_api_base,
        model_name=llm_model,
        model_kwargs={
            "stop": ["#", "##", "###", "<|end_of_text|>"],
        },
        temperature=0,
    )

    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=llm_api_base,
        model_name=llm_model,
        model_kwargs={"stop": ["."]},
    )

    generate_question = GenerateQuestionNode(
        llm=llm, template=TEMPLATE_GENERATE_QUESTION
    )
    retrieve_top1 = RetrieveNode(
        embedding_model=embedding_model,
        cache_folder=huggingface_cache_folder,
        chromadb_host=chromadb_host,
        top_k=1,
    )
    grade_retrieval = GradeRetrievalNode(
        llm=llm_grade_retrieval, template=TEMPLATE_RETRIEVAL_GRADE
    )
    update_context = UpdateContextNode(vectorstore=retrieve_top1.vectorstore)
    add_context = AddContextNode(vectorstore=retrieve_top1.vectorstore)

    retrieve = RetrieveNode(
        embedding_model=embedding_model,
        cache_folder=huggingface_cache_folder,
        chromadb_host=chromadb_host,
        top_k=top_k,
    )
    compress = CompressNode(llm=llm, template=TEMPLATE_CONTEXTUAL_COMPRESSION)
    generate = GenerateNode(llm=llm, template=TEMPLATE_GENERATE)

    workflow = StateGraph(GraphState)

    workflow.add_node("generate_question", generate_question.node)
    workflow.add_node("retrieve_top1", retrieve_top1.node)
    workflow.add_node("grade_retrieval_synthetic", grade_retrieval.node)
    workflow.add_node("update_context", update_context.node)
    workflow.add_node("add_context", add_context.node)

    workflow.add_node("retrieve", retrieve.node)
    workflow.add_node("grade_retrieval", grade_retrieval.node)
    workflow.add_node("compress", compress.node)
    workflow.add_node("generate", generate.node)
    workflow.add_node("null_generate", null_generate_node)

    workflow.set_conditional_entry_point(
        decide_to_rag, {"context": "generate_question", "rag": "retrieve"}
    )

    workflow.add_edge("generate_question", "retrieve_top1")
    workflow.add_edge("retrieve_top1", "grade_retrieval_synthetic")
    workflow.add_conditional_edges(
        "grade_retrieval_synthetic",
        decide_to_update,
        {
            "update": "update_context",
            "add": "add_context",
        },
    )
    workflow.add_edge("update_context", END)
    workflow.add_edge("add_context", END)

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

    return graph
