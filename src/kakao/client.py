import requests


class Client:
    def __init__(self, base_url):
        self.chat_url = f"{base_url}/chat"
        self.contexts_url = f"{base_url}/contexts"
        self.rag_url = f"{base_url}/rag"

    def chat(self, data):
        response = requests.post(self.chat_url, json=data)
        return response.json()

    def contexts(self, data):
        response = requests.post(self.contexts_url, json=data)
        return response.json()

    def rag(self, data):
        response = requests.post(self.rag_url, json=data)
        return response.json()
