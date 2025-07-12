from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()

# Use a small model for free CPU hosting (e.g., DistilGPT2)
chatbot = pipeline("text-generation", model="distilgpt2")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    history = data.get("history", [])
    prompt = "\n".join(history) + "\nUser: " + question
    result = chatbot(prompt, max_new_tokens=256)
    return {"answer": result[0]["generated_text"]}