from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()


HF_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        history = data.get("history", [])
        if not question:
            return {"error": "No question provided"}
        prompt = "\n".join(history) + "\nUser: " + question
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 128},
        }
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        response = requests.post(HF_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return {"answer": data[0]["generated_text"]}
            elif isinstance(data, dict) and "error" in data:
                return {"error": data["error"]}
            else:
                return {"error": "Unexpected response from Hugging Face API."}
        else:
            return {"error": f"Hugging Face API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
