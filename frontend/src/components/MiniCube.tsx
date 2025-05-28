import React, { CSSProperties } from 'react';
import { Canvas } from '@react-three/fiber';

interface Props { color: string; style?: CSSProperties; }

export default function MiniCube({ color, style }: Props) {
  return (
    <div style={{ width: 32, height: 32, ...style }}>
      <Canvas orthographic camera={{ zoom: 40, position: [2, 2, 2] }}>
        <ambientLight />
        <mesh rotation={[0.6, 0.6, 0]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color={color} />
        </mesh>
      </Canvas>
    </div>
  );
}
