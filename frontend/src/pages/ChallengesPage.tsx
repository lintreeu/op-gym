// src/pages/ChallengesPage.tsx
import React from 'react';
import { Link } from 'react-router-dom';

export default function ChallengesPage() {
  const challengeList = ['vector-addition', 'matrix-mul', 'softmax-kernel'];

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Available Challenges</h1>
      <ul className="list-disc ml-6">
        {challengeList.map((id) => (
          <li key={id}>
            <Link to={`/challenges/${id}`} className="text-blue-600 underline">
              {id.replace(/-/g, ' ')}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
