from fastapi import FastAPI, HTTPException
import json
import uvicorn
from pydantic import BaseModel

app = FastAPI()


@app.post("/")
async def root(request: dict):
    req = dict(request)
    req_text = req["userRequest"]["utterance"]

    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": f"입력하신 텍스트는 {req_text} 입니다."}}
            ]
        },
    }

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1204)
