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

class ChatRequest(BaseModel):
    messages: list
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 1500
    stream: bool = True
    user_id: str = None  # 用于记忆隔离
async def chat_completion_stream(request, user_id="default_user"):
    """流式生成器"""
    try:
        # 记忆检索
        logging.info("Start searching")
        last_user_message = next(m for m in reversed(request.messages) if m['role'] == 'user')
        if not isinstance(last_user_message['content'], str):
            relevant_memories=mem0.search(
                last_user_message['content'][0]['text'],
                user_id=user_id,
                limit=5)
        else:
            relevant_memories = mem0.search(
                query=last_user_message['content'],
                user_id=user_id,
                limit=5
            )
        logging.info("Memory searched.")
        # 构建带记忆的上下文
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)
        enhanced_messages = request.messages.copy()
        enhanced_messages.insert(-1, {"role": "system", "content": f"User Memories:\n{memories_str}"})
        
        # 调用OpenAI流式API
        stream = openai_client.chat.completions.create(
            model=request.model,
            messages=enhanced_messages,
            stream=True,
            temperature=request.temperature
        )
        logging.info("Response generating")
        full_response = ""
        for chunk in stream:
            if len(chunk.choices)>0:
                content = chunk.choices[0].delta.content or ""
                full_response += content
            yield f"data: {chunk.to_json(indent=None)}\n\n"
        # 保存完整对话到记忆
        enhanced_messages.append({"role": "assistant", "content": full_response})
        mem0.add(enhanced_messages, user_id=user_id)

    except Exception as e:
        logging.error(f"Error in streaming: {str(e)}")
        yield f"data: [ERROR] {str(e)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    try:
        # 获取或生成用户ID
        user_id = request.user_id or "defalut_user"
        logging.info("Start Handling...")
        # 流式响应处理
        if request.stream:
            return StreamingResponse(
                chat_completion_stream(request, user_id),
                media_type="text/event-stream"
            )
        
        # 非流式响应（示例实现，根据需要补充）
        raise HTTPException(501, "Non-stream mode not implemented")
    
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
