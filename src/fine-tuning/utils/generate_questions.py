import re


def batch_prompt(batch_contexts, prompt_qq_generate):
    batch_prompt_qq_generate = []
    for context in batch_contexts:
        batch_prompt_qq_generate += [prompt_qq_generate + f"정보: {context}\n"]
    return batch_prompt_qq_generate


def process_completion(batch_contexts, completion):
    batch_data = []
    for context, completion_choice in zip(batch_contexts, completion.choices):
        if completion_choice.finish_reason == "length":
            continue
        relevant_question = re.search(r"관련있는 질문:\s*(.+)", completion_choice.text)
        irrelevant_question = re.search(
            r"관련없는 질문:\s*(.+)", completion_choice.text
        )
        if relevant_question is None or irrelevant_question is None:
            continue

        data = [
            {"context": context, "question": relevant_question.group(1), "label": "예"},
            {
                "context": context,
                "question": irrelevant_question.group(1),
                "label": "아니오",
            },
        ]
        batch_data += data
    return batch_data
