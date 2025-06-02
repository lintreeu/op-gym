# Op-Gym

**Op-Gym**  是一個 GPU 記憶體存取視覺化平台，協助使用者從編輯、編譯到分析，探索 GPU kernel 在記憶體佈局、存取以及 thread/block 的配置方式。
![CUDA-Gym Overview](./assets/public/demo-overview.png)

## 🎯 目標功能

- 提供互動式 CUDA / Triton 編輯器與執行環境
- 支援 kernel 編譯後的 PTX 記憶體分析
- 模擬 blockDim / blockIdx 對記憶體佈局的影響
- 以 3D cube 呈現多個參數的記憶訪問範圍
- 支援 1D / 2D / 3D 記憶體 layout 選擇與參數調整

## ✅ 目前功能

- 編輯與執行 `kernel.cu` / `main.cu`
- 使用呼叫 [Godbolt](https://godbolt.org/) 雲端 NVCC 服務方式進行編譯
- 分析 PTX 中的 `ld.global` / `st.global` 指令與記憶體偏移
- 以 three.js 呈現多個參數記憶體的訪問 cube 視覺化
- 互動式控制 blockDim / blockIdx / 記憶體參數
- 支援 layout 模式切換（1D, row-major, col-major, 3D）

## 未來規劃

- [ ] 支援 Triton kernel 撰寫與視覺化
- [ ] 加入 LLM 編譯建議（提示、錯誤標註、自動補全）
- [ ] 自動解析main函數使用者的輸入大小
- [ ] 使用後端自架編譯環境


## 快速啟動

### 啟動後端（FastAPI）

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
````

API 文件可於：

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)


### 啟動前端（React + Vite）

```bash
cd frontend
npm install
npm run dev
```

瀏覽器打開： [http://localhost:5173](http://localhost:5173)


