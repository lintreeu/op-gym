import { useState } from 'react';
import KernelTabs, { type KernelFile } from './KernelTabs';

interface EditorPanelProps {
  onRun: (code: string) => void;
}

const defaultKernelCode = `#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Load input to shared memory
    float val = -INFINITY;
    if (tid < cols) {
        val = input[row * cols + tid];
        shared[tid] = val;
    }
    __syncthreads();

    // Compute max
    float max_val = -INFINITY;
    for (int i = 0; i < cols; ++i)
        max_val = fmaxf(max_val, shared[i]);

    // Compute exp
    float sum_exp = 0.0f;
    for (int i = 0; i < cols; ++i) {
        shared[i] = expf(shared[i] - max_val);
        sum_exp += shared[i];
    }

    // Normalize
    if (tid < cols) {
        output[row * cols + tid] = shared[tid] / sum_exp;
    }
}`;

const defaultMainCode = `
int main() {
    const int rows = 2;
    const int cols = 4;

    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 1.0f, 0.0f, -1.0f
    };
    std::vector<float> h_output(rows * cols);

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * rows * cols);
    cudaMalloc(&d_output, sizeof(float) * rows * cols);

    cudaMemcpy(d_input, h_input.data(), sizeof(float) * rows * cols, cudaMemcpyHostToDevice);

    softmax_kernel<<<rows, cols, sizeof(float) * cols>>>(d_input, d_output, cols);

    cudaMemcpy(h_output.data(), d_output, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);

    for (int r = 0; r < rows; ++r) {
        std::cout << "Row " << r << ": ";
        for (int c = 0; c < cols; ++c) {
            std::cout << h_output[r * cols + c] << " ";
        }
        std::cout << "\\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}`;

export default function EditorPanel({ defaultCode, onRun }: EditorPanelProps) {
  const [files, setFiles] = useState<KernelFile[]>([
    { name: 'kernel.cu', code: defaultKernelCode },
    { name: 'main.cu', code: defaultMainCode }
  ]);
  const [activeTab, setActiveTab] = useState('kernel.cu');
  const [log, setLog] = useState('');

  const handleRun = async () => {
    const kernelCode = files.find(f => f.name === 'kernel.cu')?.code || '';
    setLog(prev => prev + '▶️ Running kernel\n');

    const payload = {
      source_code: kernelCode,
      user_arguments: '',
      mode: 'mem',
    };

    try {
      const res = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (res.ok) {
        setLog(prev => prev + '✅ Compile finished\n');
        onRun(kernelCode);  // 通知 Playground 更新
      } else {
        throw new Error(data.detail || 'Compile error');
      }
    } catch (err: any) {
      setLog(prev => prev + `❌ ${err.message}\n`);
    }
  };

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: '#ffffff',
      borderRight: '1px solid #ddd'
    }}>
      {/* 上半部：程式碼編輯區 */}
      <div style={{ flex: 3, minHeight: 0, overflow: 'hidden' }}>
        <KernelTabs
          files={files}
          setFiles={setFiles}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />
      </div>

      {/* 下半部：控制台 */}
      <div style={{
        flex: 2,
        borderTop: '1px solid #ccc',
        padding: '1rem',
        background: '#fafafa',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h4 style={{ color: '#333', margin: 0 }}>Output / Console</h4>
          <button
            onClick={handleRun}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '0.4rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            ▶️ Run
          </button>
        </div>

        <div style={{
          flex: 1,
          overflowY: 'auto',
          marginTop: '0.75rem',
          background: '#fff',
          border: '1px solid #ccc',
          borderRadius: '6px',
          padding: '0.5rem',
          fontSize: '14px',
        }}>
          <pre style={{ whiteSpace: 'pre-wrap', margin: 0, color: '#222' }}>{log || '// No output yet'}</pre>
        </div>
      </div>
    </div>
  );
}