from pydantic import BaseModel
from typing import Dict, List, Optional


class Request(BaseModel):
    text: str = "안녕하세요"


class Response(BaseModel):
    text: str = "안녕하세요! 무엇을 도와드릴까요?"
