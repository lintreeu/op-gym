/* ──────────────────────────────────────────────
   src/App.tsx
────────────────────────────────────────────── */
import { useState } from 'react';
import KernelTabs, { type KernelFile } from './components/KernelTabs';
import RightBottomWidgets from './components/RightBottomWidgets';
import CudaBlockVisualizer from './components/CudaBlockVisualizer'

/* 預設檔案 ----------------------------- */
const defaultCode = `// flash_attention.cu

#include <cuda_runtime.h>
#include <math.h> 
#include <iostream>
#include <vector>

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int seq_len, int dim) {

    int q_idx = blockIdx.x;  // 一個 block 對應一個 query vector

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* scores = shared + dim;

    // Load Q[q_idx] 到 shared memory
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        q_shared[i] = Q[q_idx * dim + i];
    __syncthreads();

    // 計算 softmax 分母（最大值與和）
    float max_score = -1e9f;
    float sum_exp = 0.0f;

    for (int k = 0; k < seq_len; ++k) {
        float dot = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            dot += q_shared[d] * K[k * dim + d];

        // 每個 thread block 只負責一個 query，所以不需 reduction
        scores[k] = dot;
        atomicMax((int*)&max_score, __float_as_int(dot));
    }
    __syncthreads();

    for (int k = 0; k < seq_len; ++k) {
        float score = scores[k];
        score = expf(score - max_score);  // Softmax
        scores[k] = score;
        atomicAdd(&sum_exp, score);
    }
    __syncthreads();

    // Output = sum(softmax * V)
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float val = 0.0f;
        for (int k = 0; k < seq_len; ++k)
            val += (scores[k] / sum_exp) * V[k * dim + d];
        Out[q_idx * dim + d] = val;
    }
}

int main() {
    const int seq_len = 4;
    const int dim = 2;
    const int total = seq_len * dim;

    float h_Q[total] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_K[total] = {1, 0, 0, 1, 1, 1, 0, 0};
    float h_V[total] = {1, 1, 2, 2, 3, 3, 4, 4};
    float h_Out[total] = {0};

    float *d_Q, *d_K, *d_V, *d_Out;
    cudaMalloc(&d_Q, sizeof(h_Q));
    cudaMalloc(&d_K, sizeof(h_K));
    cudaMalloc(&d_V, sizeof(h_V));
    cudaMalloc(&d_Out, sizeof(h_Out));

    cudaMemcpy(d_Q, h_Q, sizeof(h_Q), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sizeof(h_K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeof(h_V), cudaMemcpyHostToDevice);

    // 每個 block 負責一個 query（seq_len 個 query）
    flash_attention_kernel<<<seq_len, dim, dim * 2 * sizeof(float)>>>(
        d_Q, d_K, d_V, d_Out, seq_len, dim);

    cudaMemcpy(h_Out, d_Out, sizeof(h_Out), cudaMemcpyDeviceToHost);

    std::cout << "Flash Attention Output:";
    for (int i = 0; i < seq_len; ++i) {
        std::cout << "Q" << i << ": ";
        for (int j = 0; j < dim; ++j)
            std::cout << h_Out[i * dim + j] << " ";
        std::cout << "";
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Out);
    return 0;
}
`;

export default function App() {
  /* ---------- state ---------- */
  const [files, setFiles] = useState<KernelFile[]>([
    { name: 'softmax.cu', code: defaultCode }
  ]);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [log, setLog] = useState('');


  /* ---------- run ---------- */
  const handleRun = async () => {
    const code = files[0].code;

    setLog(prev => prev + '▶️ Running kernel\n');

    const payload = {
      source_code: code,
      user_arguments: '',   // nvcc 旗標，如沒用到保持空字串
      mode: 'parsed' // 'ptx', 'parsed'
      // filters 可省略，或交由後端預設

    };

    try {
      const res = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (res.ok) {
        // data.ptx 為編譯後 PTX，可直接顯示或再呼叫 /ptx/parse
        setOutput(data.ptx);  // 範例：只顯示前 500 字
        setError('');
        setLog(prev => prev + '✅ Compile finished\n');
      } else {
        throw new Error(data.detail || 'Compile error');
      }
    } catch (err: any) {
      setError(err.message);
      setLog(prev => prev + `❌ ${err.message}\n`);
    }
  };

  /* ---------- layout ---------- */
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        height: '100%',
        overflow: 'hidden'
      }}
    >
      {/* ── LEFT ───────────────────────── */}
      <div
        style={{
          padding: '1rem',
          borderRight: '1px solid var(--border-color)',
          background: 'var(--bg-panel)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          height: '100%',
          minHeight: 0
        }}
      >
        {/* Editor 卡片 */}
        <div className="card" style={{ flexGrow: 1, minHeight: 0, overflow: 'auto', padding: 0 }}>
          <KernelTabs files={files} onUpdate={setFiles} />
        </div>

        {/* Output + Console 区 */}
        <div style={{ width: '100%', flexShrink: 0 }}>
          {/* header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0 }}>Output / Console</h3>
            <button onClick={handleRun} className="btn btn-blue">RUN ▶</button>
          </div>

          {/* Output box */}
          <div
            style={{
              background: 'var(--bg-card)',
              padding: '12px',
              fontFamily: 'monospace',
              whiteSpace: 'pre-wrap',
              height: '120px',
              overflowY: 'auto',
              borderRadius: 'var(--radius)',
              marginTop: '0.5rem',
              color: 'var(--text-primary)'
            }}
          >
            {error
              ? <span className="console-error">{error}</span>
              : `Output: ${output}`
            }
          </div>

          {/* Console */}
          <div
            style={{
              background: 'var(--bg-card)',
              fontFamily: 'monospace',
              padding: '0.5rem',
              borderRadius: 'var(--radius)',
              height: '160px',
              overflowY: 'auto',
              fontSize: '13px',
              marginTop: '0.5rem'
            }}
          >
            <pre className="console-log" style={{ margin: 0 }}>{log}</pre>
          </div>
        </div>
      </div>

      {/* ── RIGHT ──────────────────────── */}
      <div
        style={{
          padding: '1rem',
          background: 'var(--bg-panel)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          height: '100%',
          overflow: 'hidden'
        }}
      >
        {/* Visualizer */}
        <div style={{ flexGrow: 1, minHeight: 0, overflow: 'hidden' }}>
          <CudaBlockVisualizer/>
        </div>

        {/* Widgets */}
        <RightBottomWidgets />
      </div>
    </div>
  );
}
