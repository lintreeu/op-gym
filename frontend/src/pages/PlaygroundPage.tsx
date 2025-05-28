import React from 'react';
import { useState } from 'react';
import EditorPanel from '../components/EditorPanel';
import CudaBlockCanvasGrid from '../components/CudaBlockCanvasGrid';
import MemControlPanel from '../components/MemControlPanel';
import { type Access } from '../utils/evaluateAccessOffsets';

const defaultCode = `// flash_attention.cu\n// CUDA 程式碼預設內容...`;

const dummyAccesses: Access[] = [
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
  },
  {
    kind: "store",
    base: "arg4",
    offset: "((threadIdx.x + blockDim.x + blockIdx.x * arg5))",
    param: ["arg3", "arg5"],
    eltype: "u32",
    raw: "st.global.u32 [%rd3], %r4;"
  }
];

const dummyParams = {
  arg0: 10,
  arg1: 10,
  arg3: 10,
  arg5: 10
};



export default function PlaygroundPage() {
  const [blockDim, setBlockDim] = useState({ x: 5, y: 0 ,z: 0 });
  const [blockIdx, setBlockIdx] = useState({ x: 0, y: 0, z: 0 });
  const [basePos, setBasePos] = useState({ x: 0, y: 0, z: 0 });
  const [log, setLog] = useState('');
  const [ptx, setPtx] = useState('');
  const [accesses, setAccesses] = useState<Access[]>(dummyAccesses);
  const [params, setParams] = useState<Record<string, number>>(dummyParams);
  const [elementSizeTable, setElementSizeTable] = useState<Record<string, number>>({});
  const [activeBases, setActiveBases] = useState<string[]>(['arg0']);
  const [colors, setColors] = useState<Record<string, string>>({
    arg0: '#4da8ff',
    arg1: '#aaaaaa',
    arg3: '#ff8b4d'
  });

  const [layoutMap, setLayoutMap] = useState<Record<string, {
    layout: '1d' | 'row-major' | 'col-major' | '3d';
    dims: { rows?: number; cols?: number; depth?: number };
  }>>({});

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
    setPtx(data.ptx);
    setLog(data.ptx);
    setAccesses(data.parsed.accesses);
    setElementSizeTable(data.parsed.element_size_table);

    const usedParams = new Set(data.parsed.accesses.flatMap((a: any) => a.param));
    const paramDefaults: Record<string, number> = {};
    usedParams.forEach((p) => (paramDefaults[p] = 32));
    setParams(paramDefaults);
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', backgroundColor: '#f7f7f7' }}>
      {/* 左：編輯器 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <EditorPanel defaultCode={defaultCode} onRun={handleRun} />
      </div>

      {/* 右：可視化與控制 */}
      <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column' }}>
        {/* 上：Canvas + 控制面板（並排） */}
        <div style={{ flex: 3, display: 'flex', flexDirection: 'row', overflow: 'hidden' }}>
          <div style={{ flexBasis: '90%', maxWidth: '90%', overflow: 'hidden' }}>
            <CudaBlockCanvasGrid
              accesses={accesses.filter(a => activeBases.includes(a.base))}
              blockDim={blockDim}
              blockIdx={blockIdx}
              params={params}
              activeKind="load"
              layoutMap={layoutMap}
              colors = {colors}
              onLayoutChange={(base, layout) => {
                setLayoutMap(prev => {
                  const prevDims = prev[base]?.dims ?? {};      // 保留舊的 rows/cols/depth
                  return { ...prev, [base]: { layout, dims: prevDims } };
                });
              }}
              onDimsChange={(base, dims) => {
                setLayoutMap(prev => ({
                  ...prev,
                  [base]: {
                    ...prev[base],
                    dims: { ...prev[base]?.dims, ...dims },
                  },
                }));
              }}
              onParamsChange={(param, value) => {
                setParams(prev => ({
                  ...prev,
                  [param]: value
                }));
              }}
            />
          </div>

          {/* 控制面板右側 */}
          <div style={{
            flexBasis: '18.18%',
            maxWidth: '18.18%',
            minWidth: '240px',
            height: '100%',
            boxSizing: 'border-box',
            overflowY: 'auto',
            background: '#fff',
            margin: '2px 2px 2px 0',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          }}>
            <MemControlPanel
              blockDim={blockDim}
              setBlockDim={setBlockDim}
              blockIdx={blockIdx}
              setBlockIdx={setBlockIdx}
              base={basePos}
              setBase={setBasePos}
              params={params}
              setParams={setParams}
              activeBases={activeBases}
              setActiveBases={setActiveBases}
              colors={colors}
              setColors={setColors}
            />
          </div>
        </div>

        {/* 下方：PTX log */}
        <div style={{ flex: 2, padding: '1rem', background: '#fafafa', borderTop: '1px solid #ccc', overflowY: 'auto' }}>
          <h4 style={{ color: '#333', margin: 0 }}>Memory Access Info</h4>
          <pre style={{
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
          }}>
            {log || '(Reserved for memory access info, logs, etc.)'}
          </pre>
        </div>
      </div>
    </div>
  );
}
