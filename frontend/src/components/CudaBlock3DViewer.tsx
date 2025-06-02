import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Text } from '@react-three/drei';
import {
  evaluateAccessOffsets,
  type Access,
  type Dim3,
} from '../utils/evaluateAccessOffsets';

/* ---------- 常數 ---------- */
const CUBE_SIZE = 0.5;
const BOX_GEOM = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
const EDGE_GEOM = new THREE.EdgesGeometry(BOX_GEOM);

/* ---------- 小元件 ---------- */
function Cube({
  position, highlight, color, idx,
}: {
  position: THREE.Vector3Tuple;
  highlight: boolean;
  color: string;
  idx: number;
}) {
  return (
    <group position={position}>
      <mesh geometry={BOX_GEOM}>
        <meshStandardMaterial color={color} opacity={highlight ? 0.8 : 0.3} transparent />
      </mesh>

      {highlight && (
        <lineSegments geometry={EDGE_GEOM}>
          <lineBasicMaterial attach="material" color="#000" />
        </lineSegments>
      )}

      <Text position={[0, 0, 0]} fontSize={0.18} color="#000" anchorX="center" anchorY="middle">
        {idx}
      </Text>
    </group>
  );
}

/* ---------- 類型 ---------- */
interface Props {
  accesses: Access[];
  blockDim: Dim3;
  blockIdx: Dim3;
  params: Record<string, number>;
  colors?: Record<string, string>;           // base → hex 主色
  activeKind?: 'load' | 'store';
  layoutInfo?: {
    layout: '1d' | 'row-major' | 'col-major' | '3d-row-major' | '3d-col-major';
    dims: { rows?: number; cols?: number; depth?: number };
  };
}

/* ---------- HEX <-> HSL 工具 ---------- */
const hexToHsl = (hex: string): { h: number; s: number; l: number } => {
  hex = hex.replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16) / 255;
  const g = parseInt(hex.substring(2, 4), 16) / 255;
  const b = parseInt(hex.substring(4, 6), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0, l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h *= 60;
  }
  return { h, s: s * 100, l: l * 100 };
};

const hslToHex = (h: number, s: number, l: number): string => {
  s /= 100; l /= 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) =>
    l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const toHex = (x: number) =>
    Math.round(x * 255).toString(16).padStart(2, '0');
  return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`;
};

/* ---------- 依主色產生漸層 ---------- */
const makeGradient = (hex: string) => {
  const { h, s } = hexToHsl(hex);
  return (t: number) => hslToHex(h, s, 85 - 55 * t); // L: 85 → 30
};

/* ---------- 主元件 ---------- */
export default function CudaBlock3DViewer({
  accesses, blockDim, blockIdx, params, layoutInfo, activeKind, colors = {},
}: Props) {
  const base = accesses[0].base;
  const mainColor = colors[base] ?? '#4da8ff';
  const gradColor = makeGradient(mainColor);

  /* highlight offsets ---------------------------------------------------- */
  const baseAccessMap = useMemo(
    () => evaluateAccessOffsets(accesses, blockDim, blockIdx, params),
    [accesses, blockDim, blockIdx, params]
  );
  const activeOffsets = useMemo(() => {
    const set = baseAccessMap[base] ?? new Set<number>();
    const res = new Set<number>();
    accesses.forEach(acc => {
      baseAccessMap[acc.base]?.forEach(o => res.add(o));
    });
    return res;
  }, [accesses, baseAccessMap, activeKind, base]);

  /* layout --------------------------------------------------------------- */
  const L = layoutInfo?.layout ?? '1d';
  const R = layoutInfo?.dims?.rows ?? 1;
  const C = layoutInfo?.dims?.cols ?? 1;
  const D = layoutInfo?.dims?.depth ?? 1;
  const total = L === '1d' ? R : L.startsWith('3d') ? R * C * D : R * C;

  const toCoord = (idx: number): [number, number, number] => {
    if (L === '1d') return [idx, 0, 0];
    if (L === 'row-major') return [idx % C, Math.floor(idx / C), 0];
    if (L === 'col-major') return [Math.floor(idx / R), idx % R, 0];
    const rc = R * C, d = Math.floor(idx / rc), rem = idx % rc;
    if (L === '3d-row-major') return [rem % C, Math.floor(rem / C), d];
    return [Math.floor(rem / R), rem % R, d];        // 3d-col-major
  };

  /* spacing -------------------------------------------------------------- */
  const sp = CUBE_SIZE * 1.1;
  const maxX = L === '1d' ? R : C, maxY = L === '1d' ? 1 : R, maxZ = L.startsWith('3d') ? D : 1;
  const offX = (maxX - 1) / 2, offY = (maxY - 1) / 2, offZ = (maxZ - 1) / 2;

  /* cubes ---------------------------------------------------------------- */
  const cubes = [];
  for (let i = 0; i < total; i++) {
    const [x, y, z] = toCoord(i);
    const highlight = activeOffsets.has(i);
    const t = total > 1 ? i / (total - 1) : 0;
    cubes.push(
      <Cube
        key={i}
        idx={i}
        position={[(x - offX) * sp, (y - offY) * sp, (z - offZ) * sp]}
        highlight={highlight}
        color={gradColor(t)}
      />
    );
  }

  /* axes ----------------------------------------------------------------- */
  const axes = useMemo(() => {
    const mk = (d: THREE.Vector3, c: number) =>
      new THREE.ArrowHelper(d, new THREE.Vector3, Math.max(maxX, maxY, maxZ) * sp * 0.7, c);
    return {
      x: mk(new THREE.Vector3(1, 0, 0), 0xff0000),
      y: mk(new THREE.Vector3(0, 1, 0), 0x00ff00),
      z: mk(new THREE.Vector3(0, 0, 1), 0x0000ff)
    };
  }, [maxX, maxY, maxZ, sp]);

  /* render --------------------------------------------------------------- */
  return (
    <group>
      {cubes}
      <primitive object={axes.x} />
      <primitive object={axes.y} />
      <primitive object={axes.z} />
      <Text position={[maxX * sp * 0.8, 0, 0]} fontSize={0.25} color="#ff0000">+x</Text>
      <Text position={[0, maxY * sp * 0.8, 0]} fontSize={0.25} color="#00ff00">+y</Text>
      <Text position={[0, 0, maxZ * sp * 0.8]} fontSize={0.25} color="#0000ff">+z</Text>

      {/* 顯示 load/store label */}
      {(() => {
        const kinds = new Set(accesses.map(a => a.kind));
        const labelPos: [number, number, number] = [-(maxX * sp * 0.7), maxY * sp * 0.7, maxZ * sp * 0.6];
        const texts = [];

        if (kinds.has('load')) {
          texts.push(<Text key="load" position={labelPos} fontSize={0.25} color="limegreen" anchorX="left" anchorY="top">load</Text>);
        }
        if (kinds.has('store')) {
          texts.push(<Text key="store" position={[labelPos[0], labelPos[1] - 0.4, labelPos[2]]} fontSize={0.25} color="crimson" anchorX="left" anchorY="top">store</Text>);
        }
        return texts;
      })()}
    </group>
  );

}
