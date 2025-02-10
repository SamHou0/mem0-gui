# Dockerfile
FROM python:3.9-slim-buster

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt ./
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序代码
COPY api.py ./ 

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
