# Triton Gym

Triton Gym 是一個整合 Triton 與 React 技術的前後端互動式開發平台，透過視覺化工具幫助使用者理解並實作 GPU Kernel 的平行化設計。

## 專案架構

```
triton-gym/
├── backend/                 # FastAPI + Triton 後端
│   ├── main.py
│   ├── triton_runner.py
│   └── example/
│       └── vector_add.py
│
├── frontend/                # React + Vite 前端
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── .gitignore
├── docker-compose.yml (可選)
└── README.md
```

## 快速啟動

### 啟動後端（FastAPI）

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

後端將於 `http://localhost:8000` 啟動。

### 啟動前端（React + Vite）

```bash
cd frontend
npm install
npm run dev
```

前端將於 `http://localhost:5173` 啟動。

## Docker 快速啟動（可選）

使用 Docker 快速啟動前後端環境：

```bash
docker-compose up
```

* 前端位址：`http://localhost:5173`
* 後端位址：`http://localhost:8000`

## API 文件

後端 API 文件自動生成，透過以下網址查看：

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

## 專案說明

此專案提供 GPU kernel 開發的即時編輯與視覺化能力，結合了 Triton 的高效能 GPU 計算能力與 React 的直覺互動介面，適合開發、教學與研究使用。

## 技術堆疊

### 前端

* React
* TypeScript
* Vite
* Monaco Editor
* Radix UI
* D3.js

### 後端

* Python 3.x
* FastAPI
* Triton
* PyTorch# op-gym
