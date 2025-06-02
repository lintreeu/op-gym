import React from 'react';
import MiniCube from './MiniCube';

/* ---------- 共用小元件 ---------- */
const InputBox = ({ value, onChange }: { value: number; onChange: (v: number) => void }) => (
  <input
    type="number"
    value={value}
    onChange={e => onChange(Number(e.target.value))}
    style={{
      width: 48,
      padding: 4,
      fontSize: 13,
      border: '1px solid #ccc',
      borderRadius: 6,
      marginLeft: 8,
      textAlign: 'center',
    }}
  />
);

const RangeSlider = ({
  label,
  value,
  min = 0,
  max = 128,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) => (
  <div style={{ marginBottom: '1.5rem' }}>
    <label
      style={{
        display: 'block',
        fontSize: 14,
        fontWeight: 600,
        marginBottom: 4,
        color: '#202124',
      }}
    >
      {label}: <span style={{ color: '#5f6368' }}>{value}</span>
    </label>
    <input
      type="range"
      min={min}
      max={max}
      value={value}
      onChange={e => onChange(Number(e.target.value))}
      style={{
        width: '100%',
        boxSizing: 'border-box', // 滑桿填滿父層
        accentColor: '#1a73e8',
        height: 4,
        background: '#dadce0',
        borderRadius: 2,
      }}
    />
  </div>
);

/* ---------- Props ---------- */
interface Props {
  blockDim: { x: number; y: number; z: number };
  blockIdx: { x: number; y: number; z: number };
  setBlockDim: (v: { x: number; y: number; z: number }) => void;
  setBlockIdx: (v: { x: number; y: number; z: number }) => void;
  activeBases: string[];
  setActiveBases: (v: string[]) => void;
  colors: Record<string, string>;           // base → hex
  setColors: (c: Record<string, string>) => void;
}

/* ---------- 主元件 ---------- */
export default function MemControlPanel({
  blockDim,
  blockIdx,
  setBlockDim,
  setBlockIdx,
  activeBases,
  setActiveBases,
  colors,
  setColors,
}: Props) {
  /* 1. 線性 index ↔ xyz 轉換 */

  // 先把 0 替換成 1，避免除以 0
  const effX = blockDim.x || 1;
  const effY = blockDim.y || 1;
  const effZ = blockDim.z || 1;

  // 線性索引上限 (>=0)，若所有維度都是 0，maxLinearIdx 會是 0
  const maxLinearIdx = effX * effY * effZ - 1;

  // xyz → linear
  const currentLinear =
    blockIdx.z * effY * effX +
    blockIdx.y * effX +
    blockIdx.x;

  // linear → xyz
  const updateFromLinear = (v: number) => {
    const x = blockDim.x ? v % effX : 0;
    const y = blockDim.y ? Math.floor((v % (effX * effY)) / effX) : 0;
    const z = blockDim.z ? Math.floor(v / (effX * effY)) : 0;
    setBlockIdx({ x, y, z });
  };

  /* 2. base 名稱改由 colors 動態產生 */
  const baseList = Object.keys(colors).sort(); // 若要固定順序可移除此 sort

  return (
    <div
      style={{
        padding: '2rem',
        background: '#f8f9fa',
        overflowY: 'auto',
        overflowX: 'hidden',
        boxSizing: 'border-box',
        fontFamily: 'Arial, sans-serif',
        maxWidth: 320,
        width: '100%',
      }}
    >
      <h4 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, color: '#5f6368' }}>
        Control
      </h4>

      {/* blockDim 滑桿 */}
      <RangeSlider label="blockDim.x" value={blockDim.x} onChange={v => setBlockDim({ ...blockDim, x: v })} />
      <RangeSlider label="blockDim.y" value={blockDim.y} onChange={v => setBlockDim({ ...blockDim, y: v })} />
      <RangeSlider label="blockDim.z" value={blockDim.z} onChange={v => setBlockDim({ ...blockDim, z: v })} />

      {/* blockIdx */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: 14, fontWeight: 600, marginBottom: 4, color: '#202124' }}>
          blockIdx: ({blockIdx.x}, {blockIdx.y}, {blockIdx.z})
        </label>
        <input
          type="range"
          min={0}
          max={maxLinearIdx}
          value={currentLinear}
          onChange={e => updateFromLinear(Number(e.target.value))}
          style={{
            width: '100%',
            boxSizing: 'border-box',
            accentColor: '#1a73e8',
            height: 4,
            background: '#dadce0',
            borderRadius: 2,
          }}
        />

        {/* xyz 手動輸入 */}
        <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between' }}>
          <div>x: <InputBox value={blockIdx.x} onChange={x => setBlockIdx({ ...blockDim, x })} /></div>
          <div>y: <InputBox value={blockIdx.y} onChange={y => setBlockDim({ ...blockIdx, y })} /></div>
          <div>z: <InputBox value={blockIdx.z} onChange={z => setBlockIdx({ ...blockIdx, z })} /></div>
        </div>
      </div>

      {/* Access Bases */}
      <h4 style={{ fontSize: 14, fontWeight: 700, marginBottom: 6, color: '#5f6368' }}>Memory Access View (Max 3)</h4>
      <p style={{ fontSize: 13, marginBottom: 12 }}>Pick up to 3 memory cubes to show</p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
        {baseList.map(base => {
          const selected = activeBases.includes(base);
          const color = colors[base];
          return (
            <div key={base} style={{ textAlign: 'center', width: 60 }}>
              {/* cube 按鈕 */}
              <button
                onClick={() => {
                  if (selected) setActiveBases(activeBases.filter(b => b !== base));
                  else if (activeBases.length < 3) setActiveBases([...activeBases, base]);
                }}
                style={{
                  border: selected ? '2px solid #000' : '2px solid transparent',
                  borderRadius: 6,
                  padding: 0,
                  background: 'none',
                  cursor: 'pointer',
                  opacity: selected ? 1 : 0.4,
                  transition: 'opacity .2s',
                }}
              >
                <MiniCube color={color} />
              </button>

              <div style={{ fontSize: 12, marginTop: 2, color: '#333' }}>{base}</div>

              {/* color picker */}
              <input
                type="color"
                value={color}
                onChange={e => setColors({ ...colors, [base]: e.target.value })}
                style={{ marginTop: 4, width: 42, height: 24, border: 'none', cursor: 'pointer' }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
