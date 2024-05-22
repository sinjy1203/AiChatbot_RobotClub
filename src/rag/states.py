from langchain_core.documents import Document
from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    question: str
    generation: str
    docs: List[Document]
    contexts: str
    summarized_context: str
    retrieved: bool
    new_context: str
