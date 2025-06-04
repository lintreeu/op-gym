import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import PlaygroundPage from './PlaygroundPage';
import { type KernelFile } from '../components/KernelTabs';

export default function ChallengeDynamicPage() {
  const { challengeId } = useParams();
  const [files, setFiles] = useState<KernelFile[] | null>(null);

  useEffect(() => {
    if (!challengeId) return;

    const loadFiles = async () => {
      try {
        const kernelRes = await fetch(`/src/challenges/${challengeId}/kernel.cu`);
        const mainRes = await fetch(`/src/challenges/${challengeId}/main.cu`);
        const kernelCode = await kernelRes.text();
        const mainCode = await mainRes.text();
        setFiles([
          { name: 'kernel.cu', code: kernelCode },
          { name: 'main.cu', code: mainCode },
        ]);
      } catch (err) {
        console.error('Failed to load challenge files', err);
        setFiles(null);
      }
    };

    loadFiles();
  }, [challengeId]);

  if (!files) return <div style={{ padding: 20 }}>Loading challenge files...</div>;

  return <PlaygroundPage defaultFiles={files} />;
}
