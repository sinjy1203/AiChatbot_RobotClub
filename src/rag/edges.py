def decide_to_generate(state):
    if state["retrieved"]:
        return "generate"
    else:
        return "null_generate"


def decide_to_update(state):
    if state["retrieved"]:
        return "update"
    else:
        return "add"


def decide_to_rag(state):
    if state.get("new_context"):
        return "context"
    else:
        return "rag"
