import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

import CudaBlock3DViewer from './CudaBlock3DViewer';
import type { Access, Dim3 } from '../utils/evaluateAccessOffsets';

interface Props {
  accesses: Access[];
  blockDim: Dim3;
  blockIdx: Dim3;
  params: Record<string, number>;
  baseSize: number;
  activeKind?: 'load' | 'store';
}

export default function CudaBlockCanvasGrid(props: Props) {
  const bases = Array.from(new Set(props.accesses.map((a) => a.base)));

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '5px',
        width: '100%',
        height: '100%',
        boxSizing: 'border-box',
      }}
    >
      {bases.map((base) => {
        const filtered = props.accesses.filter((a) => a.base === base);

        return (
  <div
  key={base}
  style={{
    flex: '1 1 calc(33.33% - 16px)',
    minWidth: '240px',
    maxWidth: '400px',
    minHeight: 0,
    background: '#fff',
    borderRadius: '8px',
    boxShadow: '0 1px 1px rgba(0,0,0,0.1)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  }}
>
  {/* 上方 Canvas 區域佔 2/3 高度 */}
  <div style={{ flex: '1 1 0', position: 'relative', width: '100%' }}>
    <Canvas
      style={{ position: 'absolute', inset: 0 }}
      camera={{ position: [5, 5, 5], fov: 50 }}
    >
      <ambientLight />
      <pointLight position={[10, 10, 10]} />
      <CudaBlock3DViewer
        accesses={filtered}
        blockDim={props.blockDim}
        blockIdx={props.blockIdx}
        params={props.params}
        baseSize={props.baseSize}
        activeKind={props.activeKind}
      />
      <OrbitControls target={[0, 0, 0]} enableZoom enableRotate />
    </Canvas>
  </div>

  {/* 下方資訊欄區域佔 1/3 高度 */}
  <div
    style={{
      flex: '1 1 0',
      overflowY: 'auto',
      padding: '8px',
      fontSize: '13px',
      boxSizing: 'border-box',
    }}
  >
    <div style={{ fontWeight: 600, fontSize: '14px', color: '#333' }}>{base}</div>
    <div>
      <label style={{ fontWeight: 500 }}>Offset:</label>
      <div style={{
        fontFamily: 'monospace',
        fontSize: '12px',
        background: '#f8f8f8',
        padding: '4px 6px',
        borderRadius: '4px',
        marginTop: '2px',
      }}>
        {filtered[0].offset}
      </div>
    </div>
    <div>
      <label style={{ fontWeight: 500 }}>Params:</label>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
        {filtered[0].param.map((p) => (
          <div key={p} style={{ display: 'flex', alignItems: 'center', fontSize: '12px' }}>
            <span>{p}:</span>
            <input
              type="number"
              value={props.params[p] ?? 0}
              readOnly
              style={{
                width: '50px',
                marginLeft: '4px',
                padding: '2px 4px',
                fontSize: '12px',
                border: '1px solid #ccc',
                borderRadius: '4px',
              }}
            />
          </div>
        ))}
      </div>
    </div>
    <div>
      <label style={{ fontWeight: 500 }}>Element Type:</label>
      <div style={{ fontSize: '12px', marginTop: '2px' }}>{filtered[0].eltype}</div>
    </div>
  </div>
</div>

);

      })}
    </div>
  );
}
