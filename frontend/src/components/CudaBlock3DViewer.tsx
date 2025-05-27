import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Text } from '@react-three/drei';
import {
  evaluateAccessOffsets,
  type Access,
  type Dim3,
} from '../utils/evaluateAccessOffsets';

/* ---------- 共用幾何：整個 app 只做一次 ---------- */
const CUBE_SIZE = 0.5;
const BOX_GEOM = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
const EDGE_GEOM = new THREE.EdgesGeometry(BOX_GEOM);

/* ---------- Cube ---------- */
function Cube({
  position,
  highlight,
  color,
}: {
  position: THREE.Vector3Tuple;
  highlight: boolean;
  color: string;
}) {
  return (
    <group position={position}>
      <mesh geometry={BOX_GEOM}>
        <meshStandardMaterial
          color={highlight ? color : '#999'}
          opacity={highlight ? 1 : 0.12}
          transparent
        />
      </mesh>

      {highlight && (
        <lineSegments geometry={EDGE_GEOM}>
          <lineBasicMaterial attach="material" color="#000" />
        </lineSegments>
      )}
    </group>
  );
}

/* ---------- 主要元件 ---------- */
interface Props {
  accesses: Access[]; // 已經按 base 過濾好
  blockDim: Dim3;
  blockIdx: Dim3;
  params: Record<string, number>;
  baseSize: number;
  activeKind?: 'load' | 'store';
}
export default function CudaBlock3DViewer({
  accesses,
  blockDim,
  blockIdx,
  params,
  baseSize,
  activeKind,
}: Props) {
  /* ==== 算出哪些 offset 會被打亮 ==== */
  const baseAccessMap = useMemo(
    () => evaluateAccessOffsets(accesses, blockDim, blockIdx, params),
    [accesses, blockDim, blockIdx, params],
  );
  const activeOffsets = useMemo(() => {
    const set = baseAccessMap[accesses[0].base] ?? new Set<number>();
    const result = new Set<number>();
    accesses.forEach((acc) => {
      if (!activeKind || acc.kind === activeKind) set.forEach((o) => result.add(o));
    });
    return result;
  }, [accesses, baseAccessMap, activeKind]);

  /* ==== cube 座標 ===== */
  const gridSize = Math.ceil(Math.cbrt(baseSize));
  const offset = (gridSize - 1) / 2;
  const spacing = CUBE_SIZE * 1.1;

  const cubes = [];
  for (let idx = 0; idx < baseSize; idx++) {
    const x = idx % gridSize;
    const y = Math.floor(idx / gridSize) % gridSize;
    const z = Math.floor(idx / (gridSize * gridSize));
    const highlight = activeOffsets.has(idx);

    cubes.push(
      <Cube
        key={idx}
        position={[
          (x - offset) * spacing,
          (y - offset) * spacing,
          (z - offset) * spacing,
        ]}
        highlight={highlight}
        color={activeKind === 'store' ? '#ff8b4d' : '#4da8ff'}
      />,
    );
  }

  /* ==== 軸 ArrowHelper → 也只做一次 ==== */
  const axes = useMemo(() => {
    const mk = (dir: THREE.Vector3, color: number) =>
      new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), gridSize * spacing * 0.7, color);
    return {
      x: mk(new THREE.Vector3(1, 0, 0), 0xff0000),
      y: mk(new THREE.Vector3(0, 1, 0), 0x00ff00),
      z: mk(new THREE.Vector3(0, 0, 1), 0x0000ff),
    };
  }, [gridSize, spacing]);

  return (
    <group>
      {/* Base 名稱 */}
      <Text position={[0, gridSize * spacing * 0.55, 0]} fontSize={0.4} color="#111">
        {accesses[0].base}
      </Text>

      {/* 所有 cubes */}
      {cubes}

      {/* 三軸 */}
      <primitive object={axes.x} />
      <primitive object={axes.y} />
      <primitive object={axes.z} />

      <Text position={[gridSize * spacing * 0.78, 0, 0]} fontSize={0.25} color="#ff0000">
        +x
      </Text>
      <Text position={[0, gridSize * spacing * 0.78, 0]} fontSize={0.25} color="#00ff00">
        +y
      </Text>
      <Text position={[0, 0, gridSize * spacing * 0.78]} fontSize={0.25} color="#0000ff">
        +z
      </Text>
    </group>
  );
}
