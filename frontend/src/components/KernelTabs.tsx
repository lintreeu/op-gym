import { Tabs, TabsList, TabsTrigger, TabsContent } from '@radix-ui/react-tabs';
import Editor from '@monaco-editor/react';
import { useState } from 'react';

export type KernelFile = {
  name: string;
  code: string;
};

interface KernelTabsProps {
  files: KernelFile[];
  setFiles: (files: KernelFile[]) => void;
}

export default function KernelTabs({ files, setFiles }: KernelTabsProps) {
  const [selected, setSelected] = useState(files[0]?.name || '');

  const updateFile = (index: number, newCode: string) => {
    const updated = [...files];
    updated[index] = { ...updated[index], code: newCode };
    setFiles(updated);
  };

  const getLanguage = (fileName: string): string => {
    if (fileName.endsWith('.cu')) return 'cuda'; // Optional: map to cpp if cuda is unsupported
    if (fileName.endsWith('.cpp') || fileName.endsWith('.cc') || fileName.endsWith('.cxx')) return 'cpp';
    if (fileName.endsWith('.c')) return 'c';
    return 'plaintext';
  };

  return (
    <Tabs
      value={selected}
      onValueChange={setSelected}
      style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
    >
      {/* Tab 標籤列 */}
      <TabsList
        style={{
          padding: '0.5rem',
          background: '#f0f0f0',
          borderBottom: '1px solid #ccc',
          display: 'flex',
        }}
      >
        {files.map((file) => (
          <TabsTrigger
            key={file.name}
            value={file.name}
            style={{
              padding: '0.3rem 0.75rem',
              marginRight: '0.5rem',
              borderRadius: '4px',
              border: selected === file.name ? '1px solid #999' : '1px solid transparent',
              background: selected === file.name ? '#fff' : 'transparent',
              color: '#333',
              fontWeight: 500,
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            {file.name}
          </TabsTrigger>
        ))}
      </TabsList>

      {/* Tab 編輯器區域 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {files.map((file, index) => (
          <TabsContent
            key={file.name}
            value={file.name}
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              minHeight: 0,
            }}
          >
            <div
              style={{
                flex: 1,
                minHeight: 0,
                overflow: 'hidden',
              }}
            >
              <Editor
                height="100%"
                language={getLanguage(file.name)}
                value={file.code}
                theme="vs-light"
                options={{
                  fontSize: 14,
                  minimap: { enabled: false },
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                }}
                onMount={(editor) => {
                  editor.getDomNode()?.style.setProperty('outline', 'none');
                }}
                onChange={(value) => updateFile(index, value ?? '')}
              />
            </div>
          </TabsContent>
        ))}
      </div>
    </Tabs>
  );
}
