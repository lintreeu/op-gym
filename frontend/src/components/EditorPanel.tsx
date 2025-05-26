import { useState } from 'react';
import KernelTabs, { type KernelFile } from './KernelTabs';

interface EditorPanelProps {
  defaultCode: string;
}

export default function EditorPanel({ defaultCode }: EditorPanelProps) {
  const [files, setFiles] = useState<KernelFile[]>([
    { name: 'softmax.cu', code: defaultCode }
  ]);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [log, setLog] = useState('');

  const handleRun = async () => {
    const code = files[0].code;

    setLog((prev) => prev + '▶️ Running kernel\n');

    const payload = {
      source_code: code,
      user_arguments: '',
      mode: 'parsed',
    };

    try {
      const res = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (res.ok) {
        setOutput(data.ptx);
        setError('');
        setLog((prev) => prev + '✅ Compile finished\n');
      } else {
        throw new Error(data.detail || 'Compile error');
      }
    } catch (err: any) {
      setError(err.message);
      setLog((prev) => prev + `❌ ${err.message}\n`);
    }
  };

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: '#ffffff',
        borderRight: '1px solid #ddd'
      }}
    >
      {/* 上半部：撐滿剩餘高度 */}
      <div style={{ flex: 3, minHeight: 0, overflow: 'hidden' }}>
        <KernelTabs files={files} setFiles={setFiles} />
      </div>

      {/* 下半部：Output / Console（高度與右下固定） */}
      <div
        style={{
          flex: 2,
          borderTop: '1px solid #ccc',
          padding: '1rem',
          background: '#fafafa',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0, // ✅ 防止內容撐大
        }}
      >
        {/* 標題 + Run */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h4 style={{ color: '#333', margin: 0 }}>Output / Console</h4>
          <button
            onClick={handleRun}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '0.4rem 1rem',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            ▶️ Run
          </button>
        </div>

        {/* Output scrollable 區 */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          marginTop: '0.75rem',
          background: '#fff',
          border: '1px solid #ccc',
          borderRadius: '6px',
          padding: '0.5rem',
          fontSize: '14px',
        }}>
          <pre
            style={{
              whiteSpace: 'pre-wrap',
              margin: 0,
              color: '#222'
            }}
          >
            {output || error || '// No output yet'}
          </pre>
        </div>
      </div>
    </div>
  );
}
