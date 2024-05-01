from locust import HttpUser, task, between, TaskSet

request_json = {
    "intent": {"id": "z0tfx9xz46nwd48l7pi3eexd", "name": "블록 이름"},
    "userRequest": {
        "timezone": "Asia/Seoul",
        "params": {"ignoreMe": "true"},
        "block": {"id": "z0tfx9xz46nwd48l7pi3eexd", "name": "블록 이름"},
        "utterance": "/데이터입력 안뇽",
        "lang": "null",
        "user": {"id": "833131", "type": "accountId", "properties": {}},
    },
    "bot": {"id": "6627e49e143951673d7a23fb", "name": "봇 이름"},
    "action": {
        "name": "p264yjchec",
        "clientExtra": "null",
        "params": {},
        "id": "vmdo3yzbxro1obogrwcnc080",
        "detailParams": {},
    },
}


# class MainBehavior(TaskSet):
#     @task(weight=1)
#     def add_data_request(self):
#         request_json["userRequest"]["utterance"] = "/데이터입력 안녕하세요"
#         self.client.post("/", json=request_json, name="add data request")

#     @task(weight=10)
#     def rag_request(self):
#         request_json["userRequest"]["utterance"] = "로보트연구회가 뭐야?"
#         self.client.post("/", json=request_json, name="rag request")


class MainBehavior(TaskSet):
    @task
    def chat_request(self):
        request_json["userRequest"]["utterance"] = "인하대학교의 위치를 알려줘"
        self.client.post("/chat", json=request_json, name="chat request")


class LocustUser(HttpUser):
    tasks = [MainBehavior]
    wait_time = between(2, 10)
