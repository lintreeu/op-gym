import React, { useMemo } from 'react';
import * as THREE from 'three';
import { evaluateAccessOffsets, type Access, type Dim3 } from '../utils/evaluateAccessOffsets';

function Cube({
  x, y, z,
  highlight = false,
  color = '#4da8ff',
  size = 0.5,
}: {
  x: number;
  y: number;
  z: number;
  highlight?: boolean;
  color?: string;
  size?: number;
}) {
  const edgeGeom = new THREE.EdgesGeometry(new THREE.BoxGeometry(size, size, size));

  return (
    <group position={[x, y, z]}>
      <mesh>
        <boxGeometry args={[size, size, size]} />
        <meshStandardMaterial
          color={highlight ? color : '#999'}
          opacity={highlight ? 1.0 : 0.12}
          transparent
        />
      </mesh>
      {highlight && (
        <lineSegments geometry={edgeGeom}>
          <lineBasicMaterial attach="material" color="#000" />
        </lineSegments>
      )}
    </group>
  );
}

interface Props {
  accesses: Access[];
  blockDim: Dim3;
  blockIdx: Dim3;
  params: Record<string, number>;
  baseSize: number;
  activeKind?: 'load' | 'store'; // optional filter
}

export default function CudaBlock3DViewer({
  accesses,
  blockDim,
  blockIdx,
  params,
  baseSize,
  activeKind,
}: Props) {
  console.log(blockDim)
  console.log(blockIdx)
  const baseAccessMap = useMemo(
    () => evaluateAccessOffsets(accesses, blockDim, blockIdx, params),
    [accesses, blockDim, blockIdx, params]
  );

  // 所有 offset 被訪問的 base → offset set
  const allOffsets = useMemo(() => {
    const result = new Set<number>();
    for (const acc of accesses) {
      if (!activeKind || acc.kind === activeKind) {
        const base = acc.base;
        const set = baseAccessMap[base];
        if (set) {
          set.forEach((o) => result.add(o));
        }
      }
    }
    return result;
  }, [accesses, baseAccessMap, activeKind]);

  const gridSize = Math.ceil(Math.cbrt(baseSize));
  const cubeSize = 0.5;
  const offset = (gridSize - 1) / 2;

  const cubes = [];
  console.log(baseSize)
  console.log(gridSize)
  for (let idx = 0; idx < baseSize; idx++) {
    const x = idx % gridSize;
    const y = Math.floor(idx / gridSize) % gridSize;
    const z = Math.floor(idx / (gridSize * gridSize));
    const isBlock = allOffsets.has(idx);

    cubes.push(
      <Cube
        key={`${x}-${y}-${z}`}
        x={(x - offset) * cubeSize * 1.1}
        y={(y - offset) * cubeSize * 1.1}
        z={(z - offset) * cubeSize * 1.1}
        highlight={isBlock}
        color={activeKind === 'store' ? '#ff8b4d' : '#4da8ff'}
        size={cubeSize}
      />
    );
  }

  return <group position={[0, 0, 0]}>{cubes}</group>;
}
