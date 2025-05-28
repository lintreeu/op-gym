// src/pages/ChallengeDynamicPage.tsx
import React from 'react';
import { useParams } from 'react-router-dom';

export default function ChallengeDynamicPage() {
  const { challengeId } = useParams();

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold">Challenge: {challengeId}</h2>
      <p>展示此挑戰對應的說明與編輯器，未來可根據 `{challengeId}` 載入不同預設碼或題目內容。</p>
    </div>
  );
}
