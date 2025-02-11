from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from mem0 import Memory
import os
import logging
import json

# Initialize openai client
openai_client = OpenAI()
openai_client.api_key = os.getenv("OPENAI_API_KEY")
openai_client.base_url = os.getenv("OPENAI_BASE_URL")

# Initizalize mem0 config
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("MEM0_BASE_MODEl"),
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
            "model": os.getenv("MEM0_EMBEDDER_MODEl"),
            "embedding_dims": 3072,
            "openai_base_url": os.getenv("OPENAI_BASE_URL")
        }
    }
}

mem0 = Memory.from_config(config)
app = FastAPI()

class MemoryRequest(BaseModel):
    """The request model"""
    message: str
    user_id: str = "default_user"

@app.post("/memory/add")
async def memory_add(request:MemoryRequest):
    """Add memory to mem0"""
    try:
        mem0.add(request.message, user_id=request.user_id)
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        raise HTTPException(500, str(e))
@app.post("/memory/search")
async def memory_search(request:MemoryRequest):
    """Search memory with mem0"""
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
