import React from 'react';

const InputBox = ({ value, onChange }: { value: number; onChange: (v: number) => void }) => (
  <input
    type="number"
    value={value}
    onChange={e => onChange(Number(e.target.value))}
    style={{
      width: '48px',
      padding: '4px',
      fontSize: '13px',
      border: '1px solid #ccc',
      borderRadius: '6px',
      marginLeft: '0.5rem',
      textAlign: 'center'
    }}
  />
);

const RangeSlider = ({
  label,
  value,
  min = 0,
  max = 128,
  onChange
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
        fontSize: '14px',
        fontWeight: 600,
        marginBottom: '4px',
        color: '#202124'
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
        accentColor: '#1a73e8',
        height: '4px',
        backgroundColor: '#dadce0',
        borderRadius: '2px'
      }}
    />
  </div>
);

interface Props {
  blockDim: { x: number; y: number; z: number };
  blockIdx: { x: number; y: number; z: number };
  base: { x: number; y: number; z: number };
  setBlockDim: (v: { x: number; y: number; z: number }) => void;
  setBlockIdx: (v: { x: number; y: number; z: number }) => void;
  setBase: (v: { x: number; y: number; z: number }) => void;
  params?: Record<string, number>;
  setParams?: (p: Record<string, number>) => void;
  activeBases: string[];
  setActiveBases: (v: string[]) => void;
}

export default function MemControlPanel({
  blockDim, blockIdx, base,
  setBlockDim, setBlockIdx, setBase,
  params = {},
  setParams = () => {},
  activeBases,
  setActiveBases
}: Props) {
  const maxLinearIdx = blockDim.x * blockDim.y * blockDim.z - 1;
  const currentLinear =
    blockIdx.z * blockDim.y * blockDim.x +
    blockIdx.y * blockDim.x +
    blockIdx.x;

  const updateFromLinear = (linear: number) => {
    const x = linear % blockDim.x;
    const y = Math.floor((linear % (blockDim.x * blockDim.y)) / blockDim.x);
    const z = Math.floor(linear / (blockDim.x * blockDim.y));
    setBlockIdx({ x, y, z });
  };

  // icon mapping
  const baseIcons: Record<string, string> = {
    arg0: '/images/arg0.png',
    arg1: '/images/arg1.png',
    arg3: '/images/arg3.png',
  };

  return (
    <div
      style={{
        padding: '1.5rem',
        background: '#f8f9fa',
        boxSizing: 'border-box',
        overflowY: 'auto',
        height: '100%',
        fontFamily: 'Arial, sans-serif',
        width: '100%',
        maxWidth: '320px'
      }}
    >
      <h4 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '1rem', color: '#5f6368' }}>
        Control
      </h4>

      <RangeSlider label="blockDim.x" value={blockDim.x} onChange={v => setBlockDim({ ...blockDim, x: v })} />
      <RangeSlider label="blockDim.y" value={blockDim.y} onChange={v => setBlockDim({ ...blockDim, y: v })} />
      <RangeSlider label="blockDim.z" value={blockDim.z} onChange={v => setBlockDim({ ...blockDim, z: v })} />

      <div style={{ marginBottom: '1.5rem' }}>
        <label
          style={{
            display: 'block',
            fontSize: '14px',
            fontWeight: 600,
            marginBottom: '4px',
            color: '#202124'
          }}
        >
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
            accentColor: '#1a73e8',
            height: '4px',
            backgroundColor: '#dadce0',
            borderRadius: '2px'
          }}
        />
        <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
          <div>
            x: <InputBox value={blockIdx.x} onChange={x => setBlockIdx({ ...blockIdx, x })} />
          </div>
          <div>
            y: <InputBox value={blockIdx.y} onChange={y => setBlockIdx({ ...blockIdx, y })} />
          </div>
          <div>
            z: <InputBox value={blockIdx.z} onChange={z => setBlockIdx({ ...blockIdx, z })} />
          </div>
        </div>
      </div>

      <h4 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '0.5rem', color: '#5f6368' }}>
        Access Bases (Max 3)
      </h4>
      <p style={{ fontSize: '13px', marginBottom: '0.75rem' }}>
        Select up to 3 memory bases to visualize:
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.5rem' }}>
        {Object.entries(baseIcons).map(([base, icon]) => {
          const selected = activeBases.includes(base);
          return (
            <button
              key={base}
              onClick={() => {
                if (selected) {
                  setActiveBases(activeBases.filter(b => b !== base));
                } else if (activeBases.length < 3) {
                  setActiveBases([...activeBases, base]);
                }
              }}
              style={{
                border: selected ? '2px solid black' : '2px solid transparent',
                borderRadius: '6px',
                padding: 0,
                cursor: 'pointer',
                background: 'none'
              }}
            >
              <img
                src={icon}
                alt={base}
                title={base}
                style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '4px',
                  opacity: selected ? 1 : 0.4,
                  transition: 'opacity 0.2s ease'
                }}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}
