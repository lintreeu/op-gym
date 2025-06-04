import { Tabs, TabsList, TabsTrigger, TabsContent } from '@radix-ui/react-tabs';
import Editor from '@monaco-editor/react';
import { useEffect} from 'react';

export type KernelFile = {
  name: string;
  code: string;
};

interface KernelTabsProps {
  files: KernelFile[];
  setFiles: (files: KernelFile[]) => void;
  activeTab: string;
  setActiveTab: (name: string) => void;
}

export default function KernelTabs({ files, setFiles, activeTab, setActiveTab }: KernelTabsProps) {
  // 同步當 files 改變時預設到第一個檔案
  useEffect(() => {
    if (files.length > 0 && !files.some(f => f.name === activeTab)) {
      setActiveTab(files[0].name);
    }
  }, [files, activeTab, setActiveTab]);

  const updateFile = (index: number, newCode: string) => {
    const updated = [...files];
    updated[index] = { ...updated[index], code: newCode };
    setFiles(updated);
  };

  const getLanguage = (fileName: string): string => {
    if (fileName.endsWith('.cu')) return 'cpp';
    if (fileName.endsWith('.ptx')) return 'asm'; // PTX 可視為組合語言
    if (fileName.endsWith('.cpp') || fileName.endsWith('.cc') || fileName.endsWith('.cxx')) return 'cpp';
    if (fileName.endsWith('.c')) return 'c';
    return 'plaintext';
  };

  return (
    <Tabs
      value={activeTab}
      onValueChange={setActiveTab}
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
              border: activeTab === file.name ? '1px solid #999' : '1px solid transparent',
              background: activeTab === file.name ? '#fff' : 'transparent',
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

      {/* 編輯器內容 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {files.map((file, index) => (
       <TabsContent
  key={file.name}
  value={file.name}
  style={{
    flex: 1,
    minHeight: 0,
    overflow: 'hidden',
    display: activeTab === file.name ? 'flex' : 'none', // 這一行可避免不必要的初始化延遲
    flexDirection: 'column',
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
      width="100%" // ⭐️ 加這行可以修復部分情況下空白出現
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
