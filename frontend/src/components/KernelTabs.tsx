import Editor from '@monaco-editor/react';

export default function KernelTabs(props: { files: KernelFile[]; onUpdate: any }) {
  const file   = props.files[0];

  return (
    <Editor
      value={file.code}
      language="python"
      theme="vs-dark"          /* ← 關鍵：使用 Monaco 內建深色主題 */
      onChange={value => props.onUpdate([{ ...file, code: value || '' }])}
      options={{
        fontSize: 14,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        automaticLayout: true
      }}
    />
  );
}
