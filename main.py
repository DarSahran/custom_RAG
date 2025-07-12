from fastapi import FastAPI, Request
import requests

app = FastAPI()


HF_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"

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
        response = requests.post(HF_API_URL, json=payload)
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