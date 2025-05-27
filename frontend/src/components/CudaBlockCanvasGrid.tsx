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
  // 依 base（arg0 / arg1 / …）分組
  const bases = Array.from(new Set(props.accesses.map((a) => a.base)));

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gridTemplateRows: 'repeat(2, 1fr)',
        gap: '10px',
        height: '100%',
        width: '100%',
        padding: '10px',
        boxSizing: 'border-box',
      }}
    >
      {bases.map((base) => {
        const filtered = props.accesses.filter((a) => a.base === base);

        return (
          <div key={base} style={{ position: 'relative', width: '100%', height: '100%' }}>
            {/* 讓格子保持 1:1（方形）而且不撐開版面 */}
            <div style={{ width: '100%', paddingTop: '100%' }} />

            {/* 真正的 Canvas 以 absolute 填滿父層 */}
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
        );
      })}
    </div>
  );
}
