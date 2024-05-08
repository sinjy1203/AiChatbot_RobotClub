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

TEMPLATE_RAG = """
Answer the following question in one short sentence based only on the provided context.
Context: {context}
Human: {prompt}
Assistant:
"""

TEMPLATE_CHAT = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {prompt}
Assistant:
"""
