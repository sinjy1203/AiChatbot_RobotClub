import re
import json
from langchain_core.output_parsers import BaseOutputParser
import chromadb
from langchain_chroma import Chroma


def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


class JsonOutputParser(BaseOutputParser[dict]):
    pattern = r"\{[^{}]+\}"

    def parse(self, text: str) -> dict:
        match = re.search(self.pattern, text)
        json_string = match.group()
        # 문자열을 dictionary로 변환
        json_dict = json.loads(json_string)
        return json_dict


class BinaryOutputParser(BaseOutputParser[bytes]):
    def parse(self, text: str) -> str:
        pattern = r"(예|아니오)"
        match = re.search(pattern, text)
        return match.group(1)


def get_response_dict(response_text):
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": response_text}}]},
    }


def del_prefix(text, prefix):
    pattern = "^" + re.escape(prefix)
    return re.sub(pattern, "", text).strip()


def get_vectorstore(
    chromadb_host="vectorstore-service.default", collection_name="my_collection"
):
    client = chromadb.HttpClient(
        host=chromadb_host, settings=chromadb.Settings(allow_reset=True)
    )
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
    )
    return vectorstore
