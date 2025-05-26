import React from 'react';

const InputBox = ({ value, onChange }) => (
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

const RangeSlider = ({ label, value, min = 0, max = 128, onChange }) => (
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

export default function MemControlPanel({
  blockDim, blockIdx, basePos,
  setBlockDim, setBlockIdx, setBasePos
}) {
  // 總 Index 條（單一 slider 控制 3D blockIdx 對應的 linear index）
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
        width: '280px',
        padding: '1.5rem',
        background: '#f8f9fa',
        borderRight: '1px solid #e0e0e0',
        boxShadow: '2px 0 5px rgba(0,0,0,0.05)',
        overflowY: 'auto',
        height: '100%',
        fontFamily: 'Arial, sans-serif'
      }}
    >
      <h4 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '1rem', color: '#5f6368' }}>Control</h4>

      {/* BlockDim */}
      <RangeSlider label="blockDim.x" value={blockDim.x} onChange={v => setBlockDim({ ...blockDim, x: v })} />
      <RangeSlider label="blockDim.y" value={blockDim.y} onChange={v => setBlockDim({ ...blockDim, y: v })} />
      <RangeSlider label="blockDim.z" value={blockDim.z} onChange={v => setBlockDim({ ...blockDim, z: v })} />

      {/* blockIdx 3D 統一 slider */}
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
            x:
            <InputBox value={blockIdx.x} onChange={x => setBlockIdx({ ...blockIdx, x })} />
          </div>
          <div>
            y:
            <InputBox value={blockIdx.y} onChange={y => setBlockIdx({ ...blockIdx, y })} />
          </div>
          <div>
            z:
            <InputBox value={blockIdx.z} onChange={z => setBlockIdx({ ...blockIdx, z })} />
          </div>
        </div>
      </div>

      {/* Base Position */}
      {/* <RangeSlider label="base.x" value={basePos.x} onChange={v => setBasePos({ ...basePos, x: v })} />
      <RangeSlider label="base.y" value={basePos.y} onChange={v => setBasePos({ ...basePos, y: v })} />
      <RangeSlider label="base.z" value={basePos.z} onChange={v => setBasePos({ ...basePos, z: v })} /> */}
    </div>
  );
}
