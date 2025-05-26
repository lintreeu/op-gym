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
}

export default function MemControlPanel({
  blockDim, blockIdx, base,
  setBlockDim, setBlockIdx, setBase,
  params = {},
  setParams = () => {}
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

  return (
    <div
      style={{
        width: '200px',
        padding: '1.5rem',
        background: '#f8f9fa',
        borderRight: '1px solid #e0e0e0',
        boxShadow: '2px 0 5px rgba(0,0,0,0.05)',
        overflowY: 'auto',
        height: '100%',
        fontFamily: 'Arial, sans-serif'
      }}
    >
      <h4 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '1rem', color: '#5f6368' }}>
        Control
      </h4>

      {/* BlockDim */}
      <RangeSlider label="blockDim.x" value={blockDim.x} onChange={v => setBlockDim({ ...blockDim, x: v })} />
      <RangeSlider label="blockDim.y" value={blockDim.y} onChange={v => setBlockDim({ ...blockDim, y: v })} />
      <RangeSlider label="blockDim.z" value={blockDim.z} onChange={v => setBlockDim({ ...blockDim, z: v })} />

      {/* blockIdx slider */}
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

      {/* Param 控制區塊 */}
      <h4 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '0.5rem', color: '#5f6368' }}>
        Parameters
      </h4>
      {Object.entries(params).map(([key, value]) => (
        <div key={key} style={{ marginBottom: '0.75rem', fontSize: '13px' }}>
          <label>{key}:</label>
          <InputBox
            value={value}
            onChange={(v) => setParams({ ...params, [key]: v })}
          />
        </div>
      ))}
    </div>
  );
}
