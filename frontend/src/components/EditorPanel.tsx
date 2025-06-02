import { useState } from 'react';
import KernelTabs, { type KernelFile } from './KernelTabs';

interface EditorPanelProps {
  onRun: (kernel: string, main?: string) => void;
  filters: {
    execute: boolean;
  };
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

const Spinner = () => (
  <div style={{
    width: '16px',
    height: '16px',
    border: '2px solid #fff',
    borderTop: '2px solid #1a73e8',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  }} />
);

export default function EditorPanel({ onRun, filters }: EditorPanelProps) {
  const [files, setFiles] = useState<KernelFile[]>([
    { name: 'kernel.cu', code: defaultKernelCode },
    { name: 'main.cu', code: defaultMainCode }
  ]);
  const [activeTab, setActiveTab] = useState('kernel.cu');
  const [logLines, setLogLines] = useState<string[]>([]);
  const [stdout, setStdout] = useState('');
  const [stderr, setStderr] = useState('');
  const [isRunning, setIsRunning] = useState(false);


  const appendLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    const line = `[${timestamp}] ${message}`;
    setLogLines(prev => [...prev.slice(-9), line]); // 最多保留 10 筆
  };

  const handleRun = () => {
    const kernelCode = files.find(f => f.name === 'kernel.cu')?.code || '';
    const mainCode = files.find(f => f.name === 'main.cu')?.code || '';

    appendLog(filters.execute ? '▶️ Executing main.cu' : '▶️ Running kernel.cu');
    setStdout('');
    setStderr('');
    setIsRunning(true); // ✅ 開始執行

    const runPromise = filters.execute
      ? onRun(kernelCode, mainCode)
      : onRun(kernelCode);

    runPromise
      .then((res) => {
        setStdout(res.stdout || '');
        setStderr(res.stderr || '');
        if (res.error != '') {
          appendLog('❌ Run error');
        } else {
          appendLog('✅ Compile finished');
        }
      })
      .catch((err: any) => {
        setStdout('');
        setStderr('');
        appendLog(`❌ ${err.message || 'Run error'}`);
      })
      .finally(() => {
        setIsRunning(false); // ✅ 執行結束
      });
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
            disabled={isRunning}
            style={{
              backgroundColor: isRunning ? '#ccc' : '#1a73e8',
              color: '#fff',
              border: 'none',
              padding: '0.45rem 1.1rem',
              borderRadius: '999px',
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              fontSize: '14px',
              gap: '0.4rem',
              cursor: isRunning ? 'not-allowed' : 'pointer',
              boxShadow: '0 1px 3px rgba(0,0,0,0.15)',
              transition: 'all 0.2s ease-in-out',
            }}
            onMouseEnter={(e) => {
              if (!isRunning) {
                (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#1669d2';
                (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(-2px)';
                (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
              }
            }}
            onMouseLeave={(e) => {
              if (!isRunning) {
                (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#1a73e8';
                (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
                (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 1px 3px rgba(0,0,0,0.15)';
              }
            }}
          >
            {isRunning ? (
              <>
                <Spinner />
                <span>Running...</span>
              </>
            ) : (
              <>
                <span style={{ fontSize: '16px' }}>▶︎</span>
                <span>Run</span>
              </>
            )}
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
          <div style={{ marginBottom: '0.5rem', color: '#333' }}>
            <strong>Log:</strong>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
              {(logLines.length > 0 ? logLines : ['// No output yet']).join('\n')}
            </pre>
          </div>

          <div style={{ marginBottom: '0.5rem', color: '#555' }}>
            <strong>STDOUT:</strong>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0, color: '#222' }}>{stdout || '// (empty)'}</pre>
          </div>

          <div style={{ color: '#555' }}>
            <strong>STDERR:</strong>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0, color: '#c00' }}>{stderr || '// (empty)'}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}