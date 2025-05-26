import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

import EditorPanel from './components/EditorPanel';
import CudaBlock3DViewer from './components/CudaBlock3DViewer';
import MemControlPanel from './components/MemControlPanel';

const defaultCode = `// flash_attention.cu
// CUDA 程式碼預設內容...
`;

export default function App() {
  const [blockDim, setBlockDim] = useState({ x: 4, y: 4, z: 2 });
  const [blockIdx, setBlockIdx] = useState({ x: 0, y: 0, z: 0 });
  const [basePos, setBasePos] = useState({ x: 0, y: 0, z: 0 });
  const [log, setLog] = useState('');

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        fontFamily: 'Segoe UI, sans-serif',
        backgroundColor: '#f7f7f7',
      }}
    >
      {/* 左側區域：編輯器 + Output */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <EditorPanel defaultCode={defaultCode} />
      </div>

      {/* 右側區域：3D Viewer + 控制面板 + 記憶體資訊 */}
      <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', background: '#f5f5f5' }}>
        {/* 上半部：3D Canvas */}
        <div style={{ flex: 3, position: 'relative' }}>
          <Canvas
            camera={{ position: [6, 6, 6] }}
            style={{ background: '#f5f5f5' }}
            gl={{ preserveDrawingBuffer: true }}
          >
            <ambientLight />
            <pointLight position={[10, 10, 10]} />
            <CudaBlock3DViewer blockDim={blockDim} blockIdx={blockIdx} base={basePos} />
            <OrbitControls />
          </Canvas>

          {/* 控制面板浮動在右上 */}
          <div
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'white',
              padding: '1rem',
              borderRadius: '10px',
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
              border: '1px solid #ddd',
              zIndex: 10,
            }}
          >
            <MemControlPanel
              blockDim={blockDim}
              setBlockDim={setBlockDim}
              blockIdx={blockIdx}
              setBlockIdx={setBlockIdx}
              base={basePos}
              setBase={setBasePos}
            />
          </div>
        </div>

        {/* 下半部：記憶體訪問資訊 */}
        <div
          style={{
            flex: 2, // ✅ 與左側 Console 相同
            padding: '1rem',
            overflowY: 'auto',
            borderTop: '1px solid #ccc',
            background: '#fafafa',
          }}
        >
          <h4 style={{ color: '#333', margin: 0 }}>Memory Access Info</h4>
          <pre
            style={{
              whiteSpace: 'pre-wrap',
              color: '#444',
              background: '#fff',
              border: '1px solid #ccc',
              padding: '0.5rem',
              borderRadius: '6px',
              height: '100%',
              overflowY: 'auto',
              marginTop: '0.75rem',
              fontSize: '14px',
            }}
          >
            {log || '(Reserved for memory access info, logs, etc.)'}
          </pre>
        </div>
      </div>
    </div>
  );
}
