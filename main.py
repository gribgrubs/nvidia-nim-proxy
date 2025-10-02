from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import os
import json
from typing import AsyncGenerator

app = FastAPI()

# Get NVIDIA API key from environment variable
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        
        # Extract parameters
        messages = body.get("messages", [])
        model = body.get("model", "meta/llama-3.1-405b-instruct")
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 1024)
        top_p = body.get("top_p", 1.0)
        
        # Prepare NVIDIA NIM request
        nvidia_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                return StreamingResponse(
                    stream_nvidia_response(client, nvidia_payload, headers),
                    media_type="text/event-stream"
                )
            else:
                response = await client.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    json=nvidia_payload,
                    headers=headers
                )
                return JSONResponse(content=response.json())
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_nvidia_response(
    client: httpx.AsyncClient, 
    payload: dict, 
    headers: dict
) -> AsyncGenerator[str, None]:
    async with client.stream(
        "POST",
        f"{NVIDIA_BASE_URL}/chat/completions",
        json=payload,
        headers=headers
    ) as response:
        async for line in response.aiter_lines():
            if line.strip():
                yield f"{line}\n\n"

@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = {
        "object": "list",
        "data": [
            {
                "id": "meta/llama-3.1-405b-instruct",
                "object": "model",
                "created": 1686935002,
                "owned_by": "nvidia"
            },
            {
                "id": "meta/llama-3.1-70b-instruct",
                "object": "model",
                "created": 1686935002,
                "owned_by": "nvidia"
            },
            {
                "id": "mistralai/mixtral-8x7b-instruct-v0.1",
                "object": "model",
                "created": 1686935002,
                "owned_by": "nvidia"
            }
        ]
    }
    return JSONResponse(content=models)

@app.get("/")
async def root():
    return {"message": "OpenAI-compatible NVIDIA NIM Proxy API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
