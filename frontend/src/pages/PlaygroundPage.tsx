import React, { useState } from 'react';
import EditorPanel from '../components/EditorPanel';
import CudaBlockCanvasGrid from '../components/CudaBlockCanvasGrid';
import MemControlPanel from '../components/MemControlPanel';
import ExecuteToggle from '../components/ExecuteToggle';
import { NVCC_COMPILERS } from '../constants/NVCCVersions';
import { type Access } from '../utils/evaluateAccessOffsets';


const dummyAccesses: Access[] = [

];

const dummyParams = {

};

function hashBaseToColor(base: string): string {
  let hash = 0;
  for (let i = 0; i < base.length; i++) {
    hash = base.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 65%, 60%)`; // 較柔和的彩色調
}

export default function PlaygroundPage() {
  const [blockDim, setBlockDim] = useState({ x: 5, y: 0, z: 0 });
  const [blockIdx, setBlockIdx] = useState({ x: 0, y: 0, z: 0 });
  const [basePos, setBasePos] = useState({ x: 0, y: 0, z: 0 });
  const [log, setLog] = useState('');
  const [ptx, setPtx] = useState('');
  const [accesses, setAccesses] = useState<Access[]>(dummyAccesses);
  const [params, setParams] = useState<Record<string, number>>(dummyParams);
  const [elementSizeTable, setElementSizeTable] = useState<Record<string, number>>({});
  const [activeBases, setActiveBases] = useState<string[]>(['arg0']);
  const [colors, setColors] = useState<Record<string, string>>({});

  const [filters, setFilters] = useState({
    binary: false,
    binaryObject: false,
    execute: false,
    demangle: true,
    directives: true,
    intel: true,
    labels: true,
    commentOnly: true
  });
  const [compiler, setCompiler] = useState('nvcc128u1'); // 預設為NVCC 12.6


  const [layoutMap, setLayoutMap] = useState<Record<string, {
    layout: '1d' | 'row-major' | 'col-major' | '3d';
    dims: { rows?: number; cols?: number; depth?: number };
  }>>({});

  const handleRun = async (kernel: string, main?: string) => {
    const payload = {
      kernel_code: kernel,
      main_code: filters.execute ? main : undefined,
      mode: 'mem',
      filters,
      compiler,
    };

    // 預設配色 palette（最多 10 種 base）
    const palette = [
      '#4da8ff', '#aaaaaa', '#ff8b4d',
      '#84cc16', '#ec4899', '#f97316',
      '#8b5cf6', '#14b8a6', '#f43f5e', '#22c55e'
    ];

    try {
      const response = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      console.log('[run] result =', data);

      // ---- 錯誤處理 ----------------------------------------------------
      if (!response.ok) {
        const message = data.detail || 'Server error';
        throw new Error(message);
      }

      if (data.error && data.error !== '') {
        return {
          stdout: data.stdout || '',
          stderr: data.stderr || '',
          error: data.error,
        };
      }

      // ---- 更新 CUDA Visualization 狀態 -------------------------------
      setPtx(data.ptx);
      setAccesses(data.parsed?.accesses || []);
      setElementSizeTable(data.parsed?.element_size_table || {});

      // 更新參數參照表（如 arg0、arg1）
      const usedParams = new Set<string>();
      data.parsed?.accesses?.forEach((a: any) => a.param?.forEach((p: string) => usedParams.add(p)));
      const newParams: Record<string, number> = {};
      for (const p of usedParams) newParams[p] = 10;
      setParams(newParams);

      // 計算目前 base 列表
      const bases = Array.from(new Set(data.parsed?.accesses?.map((a: any) => a.base)));
      setActiveBases(bases.slice(0, 3));  // 預設開啟前 3 個 base 可視化

      // 配色處理：保留舊色，新的用 palette or fallback
      setColors(prev => {
        const updated: Record<string, string> = {};
        let paletteIndex = 0;
        for (const base of bases) {
          updated[base] =
            base in prev
              ? prev[base]
              : paletteIndex < palette.length
                ? palette[paletteIndex++]
                : hashBaseToColor(base);
        }
        return updated;
      });

      return {
        stdout: data.stdout || '',
        stderr: data.stderr || '',
        error: '', // 無錯誤
      };

    } catch (err: any) {
      console.error('[CUDA-Gym] Compilation or fetch failed:', err);
      return {
        stdout: '',
        stderr: '',
        error: err?.message || 'Unexpected error during run',
      };
    }
  };




  return (
    <>
      {/* Top Navigation Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.75rem 1.5rem',
        backgroundColor: '#f8f9fc',
        borderBottom: '1px solid #e0e0e0',
        fontFamily: 'sans-serif'
      }}>
        <div style={{ fontWeight: 600, fontSize: '1.1rem', color: '#202124' }}>
          <span style={{ color: '#4285F4' }}>Op</span> Gym

        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', fontSize: '0.95rem', color: '#5f6368' }}>
          <ExecuteToggle
            value={filters.execute}
            onChange={(v) => setFilters(prev => ({ ...prev, execute: v }))}
          />

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontFamily: 'sans-serif', fontSize: '14px' }}>
            <span>Compiler</span>
            <select
              value={compiler}
              onChange={e => setCompiler(e.target.value)}
              style={{
                padding: '0.35rem 0.5rem',
                borderRadius: '6px',
                border: '1px solid #ccc',
                fontSize: '0.9rem'
              }}
            >
              {Object.entries(NVCC_COMPILERS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <span>|</span>
          <a href="#" style={{ textDecoration: 'none', color: '#5f6368' }}>Documentation</a>
        </div>
      </div>

      {/* Main Layout */}
      <div style={{ display: 'flex', height: 'calc(100vh - 60px)', width: '100vw', overflow: 'hidden', backgroundColor: '#f7f7f7' }}>
        {/* Left: Editor */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <EditorPanel onRun={handleRun} filters={filters} />
        </div>

        {/* Right: Visualizer and Panel */}
        <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column' }}>
          <div style={{ flex: 3, display: 'flex', flexDirection: 'row', overflow: 'hidden' }}>
            <div style={{ flexBasis: '90%', maxWidth: '90%', overflow: 'hidden' }}>
              <CudaBlockCanvasGrid
                accesses={accesses.filter(a => activeBases.includes(a.base))}
                blockDim={blockDim}
                blockIdx={blockIdx}
                params={params}
                activeKind="load"
                layoutMap={layoutMap}
                elementSizeTable={elementSizeTable}
                colors={colors}
                onLayoutChange={(base, layout) => {
                  setLayoutMap(prev => {
                    const prevDims = prev[base]?.dims ?? {};
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

          {/* PTX Output */}
          <div style={{ flex: 2, padding: '1rem', background: '#fafafa', borderTop: '1px solid #ccc', overflowY: 'auto' }}>
            <h4 style={{ color: '#333', margin: 0 }}>Kernel PTX Output</h4>
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
              {ptx || '(Reserved for memory access info, logs, etc.)'}
            </pre>
          </div>
        </div>
      </div>
    </>
  );
}
