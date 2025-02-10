# 新建文件：api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from mem0 import Memory
import os
import logging
import json

# 初始化基础组件
openai_client = OpenAI()
openai_client.api_key = os.getenv("OPENAI_API_KEY")
openai_client.base_url = os.getenv("OPENAI_BASE_URL")

# 配置保持不变
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gemini-2.0-flash-lite-preview-02-05",
            "temperature": 0.2,
            "max_tokens": 1500,
            "openai_base_url": os.getenv("OPENAI_BASE_URL")
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.getenv("QDRANT_BASE_URL"),
            "port": 6333,
            "api_key": os.getenv("QDRANT_API_KEY"),
            "embedding_model_dims": 3072,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large",
            "embedding_dims": 3072,
            "openai_base_url": os.getenv("OPENAI_BASE_URL")
        }
    }
}

mem0 = Memory.from_config(config)
app = FastAPI()
class MemoryRequest(BaseModel):
    message: str
    user_id: str = "default_user"
@app.post("/memory/add")
async def chat_completion(request:MemoryRequest):
    try:
        mem0.add(request.message, user_id=request.user_id)
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        raise HTTPException(500, str(e))
@app.post("/memory/search")
async def chat_completion(request:MemoryRequest):
    try:
        relevant_memories=mem0.search(request.message, user_id=request.user_id)
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)
        return memories_str
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
