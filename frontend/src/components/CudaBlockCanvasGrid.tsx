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
        gap: '16px',
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
              maxWidth: '360px',
              background: '#fff',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '8px',
              boxSizing: 'border-box',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* 3D Canvas */}
            <div style={{ position: 'relative', width: '100%', paddingTop: '100%' }}>
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

            {/* 控制 bar */}
            <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '13px' }}>
              <div style={{ fontWeight: 600, fontSize: '14px', color: '#333' }}>{base}</div>

              

             
              {/* Offset formula */}
              <div>
                <label style={{ fontWeight: 500 }}>Offset:</label>
                <div style={{
                  fontFamily: 'monospace',
                  fontSize: '12px',
                  color: '#222',
                  background: '#f8f8f8',
                  padding: '4px 6px',
                  borderRadius: '4px',
                  marginTop: '2px',
                }}>
                  {filtered[0].offset}
                </div>
              </div>

              {/* Params */}
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

              {/* Eltype */}
              <div>
                <label style={{ fontWeight: 500 }}>Element Type:</label>
                <div style={{ fontSize: '12px', marginTop: '2px' }}>{filtered[0].eltype}</div>
              </div>

              {/* Raw PTX */}
              <div>
                <label style={{ fontWeight: 500 }}>PTX:</label>
                <pre style={{
                  fontSize: '11px',
                  background: '#f2f2f2',
                  padding: '6px',
                  borderRadius: '6px',
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'monospace',
                  marginTop: '4px',
                  color: '#444',
                }}>
                  {filtered[0].raw}
                </pre>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
