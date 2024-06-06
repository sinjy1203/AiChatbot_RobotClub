TEMPLATE_CHAT = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {prompt}
Assistant:
"""

# TEMPLATE_RETRIEVAL_GRADE = """
# You are a grader assessing whether the information in the context is sufficient to answer the question.
# Give a binary score "예" or "아니오" score to indicate whether the context is sufficient.
# Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
# Question: {question}
# Context: {contexts}

# Grader:
# """

TEMPLATE_RETRIEVAL_GRADE = """주어진 질문과 정보가 주어졌을 때 질문에 답하기에 충분한 정보인지 평가해줘.
정보가 충분한지를 평가하기 위해 "예" 또는 "아니오"로 답해줘. 

### 질문: 
{question}

### 정보: 
{contexts}

### 평가: 
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

TEMPLATE_GENERATE_QUESTION = """
Given the following context, generate a factual question that is relevant to the information provided in the context.
Provide question in korean and no preamble or explanation.
Context: {new_context}
Response: 
"""
