import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

import EditorPanel from './components/EditorPanel';
import CudaBlock3DViewer from './components/CudaBlock3DViewer';
import MemControlPanel from './components/MemControlPanel';
import { type Access } from './utils/evaluateAccessOffsets';

const defaultCode = `// flash_attention.cu
// CUDA 程式碼預設內容...
`;
const dummyAccesses = [
  {
    kind: "load",
    base: "arg0",
    offset: "((threadIdx.x + blockIdx.x * arg5))",
    param: ["arg0", "arg5"],
    eltype: "f32",
    raw: "ld.global.f32 %f1, [%rd1];"
  },
  {
    kind: "load",
    base: "arg1",
    offset: "(threadIdx.x * 1)",
    param: ["arg1"],
    eltype: "f32",
    raw: "ld.global.f32 %f2, [%rd2];"
  },
  {
    kind: "store",
    base: "arg3",
    offset: "((threadIdx.x + blockDim.x + blockIdx.x * arg5))",
    param: ["arg3", "arg5"],
    eltype: "u32",
    raw: "st.global.u32 [%rd3], %r4;"
  }
];

const dummyParams = {
  arg0: 0,
  arg1: 0,
  arg3: 0,
  arg5: 32 // 例如：stride or size
};

export default function App() {
  const [blockDim, setBlockDim] = useState({ x: 4, y: 4, z: 2 });
  const [blockIdx, setBlockIdx] = useState({ x: 0, y: 0, z: 0 });
  const [basePos, setBasePos] = useState({ x: 0, y: 0, z: 0 });
  const [log, setLog] = useState('');
  const [ptx, setPtx] = useState('');
  const [accesses, setAccesses] = useState<Access[]>([]);
  const [elementSizeTable, setElementSizeTable] = useState<Record<string, number>>(dummyAccesses);
  const [params, setParams] = useState<Record<string, number>>(dummyParams);
  const [baseSize, setBaseSize] = useState(512); // 記憶體總範圍（cube 長度）

  const handleRun = async (code: string) => {
    const res = await fetch('http://localhost:8000/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_code: code,
        mode: 'mem',
        filters: { demangle: true },
      }),
    });

    const data = await res.json();
    console.log("Accesses:", data.parsed.accesses);
    setPtx(data.ptx);
    setLog(data.ptx); // 顯示原始 PTX
    setAccesses(data.parsed.accesses);
    setElementSizeTable(data.parsed.element_size_table);

    // 初始化參數（如 arg0、arg5）為 32 預設值
    const usedParams = new Set(data.parsed.accesses.flatMap((a: any) => a.param));
    const paramDefaults: Record<string, number> = {};
    usedParams.forEach((p) => (paramDefaults[p] = 32));
    setParams(paramDefaults);
  };

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
      {/* 左側：CUDA 編輯器 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <EditorPanel defaultCode={defaultCode} onRun={handleRun} />
      </div>

      {/* 右側：Canvas + 控制面板 + Log */}
      <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', background: '#f5f5f5' }}>
        {/* 上：3D Canvas */}
        <div style={{ flex: 3, position: 'relative' }}>
          <Canvas
            camera={{ position: [6, 6, 6] }}
            style={{ background: '#f5f5f5' }}
            gl={{ preserveDrawingBuffer: true }}
          >
            <ambientLight />
            <pointLight position={[10, 10, 10]} />
            <CudaBlock3DViewer
              accesses={accesses}
              blockDim={blockDim}
              blockIdx={blockIdx}
              params={params}
              baseSize={baseSize}
              activeKind="load" // or "store"
            />
            <OrbitControls />
          </Canvas>

          {/* 控制面板浮動 */}
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
              params={params}
              setParams={setParams}
            />
          </div>
        </div>

        {/* 下：記憶體訪問 log 顯示 */}
        <div
          style={{
            flex: 2,
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
