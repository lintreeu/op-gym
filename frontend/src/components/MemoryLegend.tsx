// src/components/MemoryLegend.tsx

export function MemoryLegend() {
  return (
    <div style={{ display: 'flex', gap: '1rem', fontSize: '14px', marginTop: '0.5rem' }}>
      <div><span style={square('#555')} /> X來源</div>
      <div><span style={square('#aaa')} /> Y來源</div>
      <div><span style={square('#ffd700')} /> Z寫入</div>
      <div><span style={square('black')} /> 遮罩無效</div>
    </div>
  );
}

function square(color: string): React.CSSProperties {
  return {
    display: 'inline-block',
    width: '12px',
    height: '12px',
    backgroundColor: color,
    marginRight: '4px'
  };
}