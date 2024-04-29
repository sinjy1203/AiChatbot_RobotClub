from fastapi import FastAPI, HTTPException
import json
import uvicorn
from pydantic import BaseModel

import ollama

app = FastAPI()


@app.post("/")
async def root(request: dict):
    req = dict(request)
    req_text = req["userRequest"]["utterance"]
    print(req_text)
    if req_text.startswith("/데이터입력"):
        response_text = "데이터 입력이 완료하였습니다."
    else:
        response = ollama.chat(
            model="EEVE-Korean-10.8B",
            messages=[
                {
                    "role": "user",
                    "content": req_text,
                },
            ],
        )
        response_text = response["message"]["content"].strip()

    print(response_text)
    res = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": response_text}}]},
    }

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
