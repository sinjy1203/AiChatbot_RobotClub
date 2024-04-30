import re
from langchain_core.documents import Document


def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


def add_data(retriever, req_text, prefix="/데이터입력"):
    pattern = "^" + re.escape("/데이터입력")
    req_text = re.sub(pattern, "", req_text).strip()

    retriever.add_documents([Document(page_content=req_text)])
    response_text = "데이터 입력이 완료하였습니다."
    return response_text


def get_response_dict(response_text):
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": response_text}}]},
    }
