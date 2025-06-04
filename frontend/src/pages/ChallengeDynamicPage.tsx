import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import PlaygroundPage from './PlaygroundPage';
import type { KernelFile } from '../components/KernelTabs';
import { getKernelUrl, getMainUrl } from '../constants/CHALLENGES';

export default function ChallengeDynamicPage() {
  const { challengeId } = useParams<{ challengeId: string }>();
  const [files, setFiles] = useState<KernelFile[] | null>(null);

  useEffect(() => {
    async function loadChallenge() {
      try {
        const [kernelRes, mainRes] = await Promise.all([
          fetch(getKernelUrl(challengeId!)),
          fetch(getMainUrl(challengeId!))
        ]);

        if (!kernelRes.ok || !mainRes.ok) throw new Error('Challenge not found');

        const kernelCode = await kernelRes.text();
        const mainCode = await mainRes.text();

        setFiles([
          { name: 'kernel.cu', code: kernelCode },
          { name: 'main.cu', code: mainCode }
        ]);
      } catch (err) {
        console.error(err);
        setFiles(null);
      }
    }

    if (challengeId) loadChallenge();
  }, [challengeId]);

  if (!files) return <p>Loading...</p>;
  return <PlaygroundPage defaultFiles={files} />;
}
