FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000


ENV OPENAI_BASE_URL=https://api.openai.com/v1
ENV MEM0_BASE_MODEl=gpt-4o-mini
ENV MEM0_EMBEDDER_MODEl=text-embedding-3-large

ENV OPENAI_API_KEY=sk-1234567abcd
ENV QDRANT_BASE_URL=http://127.0.0.1
ENV QDRANT_API_KEY=1234567abcd

# 启动应用
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]