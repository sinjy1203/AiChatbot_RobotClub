import re
import json
from langchain_core.output_parsers import BaseOutputParser


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
    def parse(self, text: str) -> bytes:
        if "예" in text:
            return "예"
        else:
            return "아니오"
