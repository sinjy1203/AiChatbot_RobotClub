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

    response = ollama.chat(
        model="llama2",
        messages=[
            {
                "role": "user",
                "content": req_text,
            },
        ],
    )

    res = {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": response["message"]["content"]}}]
        },
    }

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
