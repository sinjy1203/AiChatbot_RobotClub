from pydantic import BaseModel
from typing import Dict, List, Optional


class UserRequest(BaseModel):
    timezone: str = "Asia/Seoul"
    params: Dict[str, str] = {"ignoreMe": "true"}
    block: Dict[str, str] = {"id": "py9qmysqdr68fn9h2oaif7it", "name": "블록 이름"}
    utterance: str = "/데이터입력 로보트연구회 회장은 홍길동이야"
    lang: Optional[str] = None
    user: Dict = {}


class Request(BaseModel):
    intent: Dict = {}
    userRequest: UserRequest
    bot: Dict = {}
    action: Dict = {}
    contexts: List = []
