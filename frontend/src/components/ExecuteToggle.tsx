import React from 'react';

interface Props {
  value: boolean;
  onChange: (v: boolean) => void;
}

export default function ExecuteToggle({ value, onChange }: Props) {
  return (
    <label
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontFamily: 'sans-serif',
        fontSize: '14px',
      }}
    >
      <span
        style={{
          color: value ? '#1a73e8' : '#5f6368',
          fontWeight: value ? 600 : 500,
        }}
      >
        {value ? 'Execute' : 'Analyze'}
      </span>
      <div
        onClick={() => onChange(!value)}
        style={{
          position: 'relative',
          width: '36px',
          height: '20px',
          borderRadius: '9999px',
          backgroundColor: value ? '#1a73e8' : '#ccc',
          cursor: 'pointer',
          transition: 'background-color 0.2s',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: '2px',
            left: value ? '18px' : '2px',
            width: '16px',
            height: '16px',
            borderRadius: '9999px',
            backgroundColor: '#fff',
            boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
            transition: 'left 0.2s',
          }}
        />
      </div>
    </label>
  );
}
