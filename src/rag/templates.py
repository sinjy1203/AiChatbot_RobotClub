# TEMPLATE_RAG = """
# Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {prompt}
# """

# TEMPLATE_RAG = """
# A chat between a curious user and an artificial intelligence assistant. The assistant answers the following question based only on the provided context
# Context: {context}
# Human: {prompt}
# Assistant:
# """

# TEMPLATE_RAG = """
# Answer the following question based only on the provided Context
# Context: {context}
# Human: {prompt}
# Assistant:
# """s

# TEMPLATE_RAG = """
# Answer the following question in one short sentence based only on the provided context.
# Context: {context}
# Human: {prompt}
# Assistant:
# """


TEMPLATE_CHAT = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {prompt}
Assistant:
"""

TEMPLATE_RETRIEVAL_GRADE = """
You are a grader assessing whether the information in the context is sufficient to answer the question.
Give a binary score "예" or "아니오" score to indicate whether the context is sufficient.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
Question: {question}
Context: {contexts}

Grader: 
"""

TEMPLATE_CONTEXTUAL_COMPRESSION = """
Given the following question and contexts, extract any part of the context *AS IS* that is relevant to answer the question. 
Remember, *DO NOT* edit the extracted parts of the context.
Question: {question}
Contexts: {contexts}
Extracted relevant parts: 
"""

TEMPLATE_GENERATE = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives answers to the user's questions based on the following context. please keep the answer brief.
Human: {question}
Context: {contexts}
Assistant: 
"""
