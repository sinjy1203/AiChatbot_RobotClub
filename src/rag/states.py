from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    question: str
    generation: str
    contexts: List[str]
    summarized_context: str
    retrieved: bool
