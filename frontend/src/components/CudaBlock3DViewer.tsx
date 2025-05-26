import React from 'react';
import * as THREE from 'three';

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
          <lineBasicMaterial attach="material" color="#000" linewidth={1} />
        </lineSegments>
      )}
    </group>
  );
}

export default function CudaBlock3DViewer({
  blockDim,
  blockIdx,
  activeKind,
}: {
  blockDim: { x: number; y: number; z: number };
  blockIdx: { x: number; y: number; z: number };
  activeKind: 'load' | 'store';
}) {
  const gridSize = 8;
  const cubeSize = 0.5;

  const cubes = [];
  const offset = (gridSize - 1) / 2; // 為使整體中心對齊

  for (let z = 0; z < gridSize; z++) {
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const isInBlock =
          x >= blockIdx.x && x < blockIdx.x + blockDim.x &&
          y >= blockIdx.y && y < blockIdx.y + blockDim.y &&
          z >= blockIdx.z && z < blockIdx.z + blockDim.z;

        cubes.push(
          <Cube
            key={`${x}-${y}-${z}`}
            x={(x - offset) * cubeSize * 1.1}
            y={(y - offset) * cubeSize * 1.1}
            z={(z - offset) * cubeSize * 1.1}
            highlight={isInBlock}
            color={activeKind === 'load' ? '#4da8ff' : '#ff8b4d'}
            size={cubeSize}
          />
        );
      }
    }
  }

  return (
    <group position={[0, 0, 0]}>
      {cubes}
    </group>
  );
}
