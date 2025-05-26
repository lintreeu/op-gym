import React, { useState } from 'react';

function safeEval(expr, context) {
  try {
    const func = new Function(...Object.keys(context), `return ${expr};`);
    return func(...Object.values(context));
  } catch {
    return null;
  }
}

export default function CudaBlockVisualizer({ memAccess, blockDim, blockIdx, basePos }) {
  const [activeTab, setActiveTab] = useState('load');
  const filteredAccess = memAccess.filter(acc => acc.kind === activeTab);

  const width = 32;
  const height = 16;
  const total = width * height;

  const highlightIndices = new Set();

  const context = {
    blockDim, blockIdx,
    threadIdx: { x: 0, y: 0, z: 0 },
    element_size_f32: 1,
    element_size_f64: 1,
    element_size_u32: 1,
    element_size_u64: 1,
    element_size_s32: 1,
    element_size_s64: 1,
    arg0: 0, arg1: 0, arg3: 0, arg5: 4
  };

  for (let tid = 0; tid < blockDim.x; tid++) {
    context.threadIdx.x = tid;

    for (const acc of filteredAccess) {
      const expr = acc.offset || '0';
      const offset = safeEval(expr, context);
      if (typeof offset === 'number') {
        highlightIndices.add(offset);
      }
    }
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Tabs */}
      <div style={{ display: 'flex', marginBottom: '0.5rem' }}>
        {['load', 'store'].map(kind => (
          <button
            key={kind}
            onClick={() => setActiveTab(kind)}
            style={{
              padding: '4px 12px',
              backgroundColor: activeTab === kind ? '#1a73e8' : '#e8eaed',
              color: activeTab === kind ? 'white' : '#444',
              border: 'none',
              marginRight: '6px',
              borderRadius: '4px',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            {kind.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Visual Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${width}, 10px)`,
          gridTemplateRows: `repeat(${height}, 10px)`,
          gap: '1px',
          background: '#ccc',
          padding: '4px',
          flexGrow: 1,
          overflow: 'auto'
        }}
      >
        {Array.from({ length: total }).map((_, idx) => {
          const globalIdx = idx + basePos.x;
          const isActive = highlightIndices.has(globalIdx);
          return (
            <div
              key={idx}
              style={{
                width: 10,
                height: 10,
                backgroundColor: isActive ? '#66c2ff' : '#f5f5f5',
                border: '1px solid #ddd',
                boxSizing: 'border-box'
              }}
              title={`offset: ${globalIdx}`}
            />
          );
        })}
      </div>
    </div>
  );
}
