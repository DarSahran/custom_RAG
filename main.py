from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()

chatbot = pipeline("text-generation", model="distilgpt2")

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        history = data.get("history", [])
        if not question:
            return {"error": "No question provided"}
        prompt = "\n".join(history) + "\nUser: " + question
        result = chatbot(prompt, max_new_tokens=256)
        return {"answer": result[0]["generated_text"]}
    except Exception as e:
        return {"error": str(e)}