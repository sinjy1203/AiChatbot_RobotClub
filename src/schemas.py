from pydantic import BaseModel
from typing import Dict


class Intent(BaseModel):
    id: str = "py9qmysqdr68fn9h2oaif7it"
    name: str = "블록 이름"


class User(BaseModel):
    id: str = "530566"
    type: str = "accountId"
    properties: Dict[str, str] = {}


class UserRequest(BaseModel):
    timezone: str = "Asia/Seoul"
    params: Dict[str, str] = {"ignoreMe": "true"}
    block: Dict[str, str] = {"id": "py9qmysqdr68fn9h2oaif7it", "name": "블록 이름"}
    utterance: str = "/데이터입력 안뇽"
    lang: str | None = None
    user: User


class Bot(BaseModel):
    id: str = "6627e49e143951673d7a23fb"
    name: str = "봇 이름"


class Action(BaseModel):
    name: str = "88eeu30mi0"
    clientExtra: str | None = None
    params: Dict = {}
    id: str = "fmnb8n78nqgphewwz7n3qu58"
    detailParams: Dict = {}


class Request(BaseModel):
    intent: Intent
    userRequest: UserRequest
    bot: Bot
    action: Action
