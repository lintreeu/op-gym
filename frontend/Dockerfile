# frontend/Dockerfile
FROM node:20

WORKDIR /app

# 複製前端程式碼
COPY frontend /app

# 安裝依賴
RUN npm install
RUN npm install -g vite

EXPOSE 5173

CMD [ "bash" ]