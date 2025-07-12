from fastapi import FastAPI, Request
import os
from huggingface_hub import InferenceClient


app = FastAPI()
client = InferenceClient(
    provider="auto",
    api_key=os.environ.get("HF_API_TOKEN"),
)



@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        history = data.get("history", [])
        if not question:
            return {"error": "No question provided"}
        messages = []
        for h in history:
            messages.append({"role": "user", "content": h})
        messages.append({"role": "user", "content": question})
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=messages,
        )
        if completion and completion.choices and completion.choices[0].message:
            return {"answer": completion.choices[0].message.content}
        else:
            return {"error": "No answer returned from model."}
    except Exception as e:
        return {"error": str(e)}
