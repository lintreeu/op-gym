// src/components/EditorPanel.tsx
import { useState } from 'react';
import KernelTabs, { type KernelFile } from './KernelTabs';

interface EditorPanelProps {
  files: KernelFile[];
  setFiles: (files: KernelFile[]) => void;
  onRun: (kernel: string, main?: string) => void;
  filters: {
    execute: boolean;
  };
}

const Spinner = () => (
  <div style={{
    width: '16px',
    height: '16px',
    border: '2px solid #fff',
    borderTop: '2px solid #1a73e8',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  }} />
);

export default function EditorPanel({ files, setFiles, onRun, filters }: EditorPanelProps) {
  const [activeTab, setActiveTab] = useState('kernel.cu');
  const [logLines, setLogLines] = useState<string[]>([]);
  const [stdout, setStdout] = useState('');
  const [stderr, setStderr] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const appendLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    const line = `[${timestamp}] ${message}`;
    setLogLines(prev => [...prev.slice(-9), line]); // 最多保留 10 筆
  };

  const handleRun = () => {
    const kernelCode = files.find(f => f.name === 'kernel.cu')?.code || '';
    const mainCode = files.find(f => f.name === 'main.cu')?.code || '';

    appendLog(filters.execute ? '▶️ Executing main.cu' : '▶️ Running kernel.cu');
    setStdout('');
    setStderr('');
    setIsRunning(true);

    const runPromise = filters.execute
      ? onRun(kernelCode, mainCode)
      : onRun(kernelCode);

    runPromise
      .then((res) => {
        setStdout(res.stdout || '');
        setStderr(res.stderr || '');
        if (res.error !== '') {
          appendLog('❌ Run error');
        } else {
          appendLog('✅ Compile finished');
        }
      })
      .catch((err: any) => {
        setStdout('');
        setStderr('');
        appendLog(`❌ ${err.message || 'Run error'}`);
      })
      .finally(() => {
        setIsRunning(false);
      });
  };

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: '#ffffff',
      borderRight: '1px solid #ddd'
    }}>
      {/* 上半部：程式碼編輯區 */}
      <div style={{ flex: 3, minHeight: 0, overflow: 'hidden' }}>
        <KernelTabs
          files={files}
          setFiles={setFiles}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />
      </div>

      {/* 下半部：控制台 */}
      <div style={{
        flex: 2,
        borderTop: '1px solid #ccc',
        padding: '1rem',
        background: '#fafafa',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h4 style={{ color: '#333', margin: 0 }}>Output / Console</h4>
          <button
            onClick={handleRun}
            disabled={isRunning}
            style={{
              backgroundColor: isRunning ? '#ccc' : '#1a73e8',
              color: '#fff',
              border: 'none',
              padding: '0.45rem 1.1rem',
              borderRadius: '999px',
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              fontSize: '14px',
              gap: '0.4rem',
              cursor: isRunning ? 'not-allowed' : 'pointer',
              boxShadow: '0 1px 3px rgba(0,0,0,0.15)',
              transition: 'all 0.2s ease-in-out',
            }}
            onMouseEnter={(e) => {
              if (!isRunning) {
                const btn = e.currentTarget as HTMLButtonElement;
                btn.style.backgroundColor = '#1669d2';
                btn.style.transform = 'translateY(-2px)';
                btn.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
              }
            }}
            onMouseLeave={(e) => {
              if (!isRunning) {
                const btn = e.currentTarget as HTMLButtonElement;
                btn.style.backgroundColor = '#1a73e8';
                btn.style.transform = 'translateY(0)';
                btn.style.boxShadow = '0 1px 3px rgba(0,0,0,0.15)';
              }
            }}
          >
            {isRunning ? (
              <>
                <Spinner />
                <span>Running...</span>
              </>
            ) : (
              <>
                <span style={{ fontSize: '16px' }}>▶︎</span>
                <span>Run</span>
              </>
            )}
          </button>
        </div>

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
          <div style={{ marginBottom: '0.5rem', color: '#333' }}>
            <strong>Log:</strong>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
              {(logLines.length > 0 ? logLines : ['']).join('\n')}
            </pre>
          </div>

          <div style={{ marginBottom: '0.5rem', color: '#555' }}>
            <strong>STDOUT:</strong>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0, color: '#222' }}>{stdout || ''}</pre>
          </div>

          <div style={{ color: '#555' }}>
            <strong>STDERR:</strong>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0, color: '#c00' }}>{stderr || ''}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
