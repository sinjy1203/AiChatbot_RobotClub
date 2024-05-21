def decide_to_generate(state):
    if state["retrieved"]:
        return "generate"
    else:
        return "not_generate"
