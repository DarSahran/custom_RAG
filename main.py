from fastapi import FastAPI, Request
from transformers import pipeline

app = FastAPI()

# Use a small model for free CPU hosting (e.g., TinyLlama, DistilGPT2, Phi-2)
chatbot = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    history = data.get("history", [])
    prompt = "\n".join(history) + "\nUser: " + question
    result = chatbot(prompt, max_new_tokens=256)
    return {"answer": result[0]["generated_text"]}