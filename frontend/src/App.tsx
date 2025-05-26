import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import KernelTabs, { type KernelFile } from './components/KernelTabs';
import CudaBlock3DViewer from './components/CudaBlock3DViewer';
import MemControlPanel from './components/MemControlPanel';

const defaultCode = `// flash_attention.cu
// 省略 CUDA 原始碼，保留為 defaultCode 內容
`;

export default function App() {
  const [files, setFiles] = useState<KernelFile[]>([
    { name: 'softmax.cu', code: defaultCode }
  ]);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [log, setLog] = useState('');
  const [blockDim, setBlockDim] = useState({ x: 4, y: 4, z: 2 });
  const [blockIdx, setBlockIdx] = useState({ x: 0, y: 0, z: 0 });
  const [basePos, setBasePos] = useState({ x: 0, y: 0, z: 0 });
  const [activeTab, setActiveTab] = useState<'load' | 'store'>('load');

  const handleRun = async () => {
    const code = files[0].code;
    setLog(prev => prev + '▶️ Running kernel\n');

    const payload = {
      source_code: code,
      user_arguments: '',
      mode: 'parsed'
    };

    try {
      const res = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (res.ok) {
        setOutput(data.ptx);
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

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', height: '100vh' }}>
      {/* LEFT: 原本 UI */}
      <div style={{
        padding: '1rem',
        borderRight: '1px solid #ddd',
        background: '#f6f6f6',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        height: '100%',
        minHeight: 0
      }}>
        <div className="card" style={{ flexGrow: 1, minHeight: 0, overflow: 'auto', padding: 0 }}>
          <KernelTabs files={files} onUpdate={setFiles} />
        </div>

        <div style={{ width: '100%', flexShrink: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0 }}>Output / Console</h3>
            <button onClick={handleRun} className="btn btn-blue">RUN ▶</button>
          </div>

          <div style={{
            background: '#fff',
            padding: '12px',
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            height: '120px',
            overflowY: 'auto',
            borderRadius: '8px',
            marginTop: '0.5rem',
            color: '#333'
          }}>
            {error ? <span style={{ color: 'red' }}>{error}</span> : `Output: ${output}`}
          </div>

          <div style={{
            background: '#fff',
            fontFamily: 'monospace',
            padding: '0.5rem',
            borderRadius: '8px',
            height: '160px',
            overflowY: 'auto',
            fontSize: '13px',
            marginTop: '0.5rem'
          }}>
            <pre style={{ margin: 0 }}>{log}</pre>
          </div>
        </div>
      </div>

      {/* RIGHT: 視覺化 + 控制 */}

      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* 上方視覺化區塊：佔 60% */}
        <div style={{ flexBasis: '60%', display: 'flex', flexDirection: 'row' }}>
          <div style={{ flexGrow: 1, background: '#eaeaea' }}>
            <Canvas camera={{ position: [12, 12, 12], fov: 50 }}>
              <ambientLight />
              <directionalLight position={[10, 10, 10]} />
              <OrbitControls />
              <CudaBlock3DViewer
                blockDim={blockDim}
                blockIdx={blockIdx}
                activeKind={activeTab}
              />
            </Canvas>
          </div>

          <div style={{ width: '280px', background: '#f8f9fa', display: 'flex', flexDirection: 'column' }}>
            <MemControlPanel
              blockDim={blockDim}
              blockIdx={blockIdx}
              basePos={basePos}
              setBlockDim={setBlockDim}
              setBlockIdx={setBlockIdx}
              setBasePos={setBasePos}
            />
          </div>
        </div>

        {/* 預留下方空間（佔 40%） */}
        <div style={{
          flexBasis: '40%',
          background: '#fafafa',
          borderTop: '1px solid #ccc',
          padding: '1rem',
          fontStyle: 'italic',
          color: '#888'
        }}>
          {/* 可放日後的視覺化說明 / 詳細資訊 */}
          (Reserved for memory access info, logs, etc.)
        </div>
      </div>

    </div>
  );
}
