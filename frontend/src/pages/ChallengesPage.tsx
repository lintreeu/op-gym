import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { METADATA_URL } from '../constants/CHALLENGES';

interface ChallengeMeta {
  id: string;
  title: string;
  difficulty: string;
  tags: string[];
}


export default function ChallengesPage() {
  const [challenges, setChallenges] = useState<ChallengeMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    async function fetchMetadata() {
      try {
        const res = await fetch(METADATA_URL);
        const data = await res.json();
        setChallenges(data);
      } catch (err) {
        console.error('Failed to fetch metadata:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchMetadata();
  }, []);

  if (loading) return <p>Loading challenges...</p>;

  return (
    <div style={{ padding: '1rem' }}>
      <h1 style={{ marginBottom: '1rem' }}>Op-Gym Challenges</h1>
      {challenges.map((challenge) => (
        <div
          key={challenge.id}
          onClick={() => navigate(`/challenges/${challenge.id}`)}
          style={{
            cursor: 'pointer',
            padding: '1rem',
            marginBottom: '0.75rem',
            border: '1px solid #ccc',
            borderRadius: '8px'
          }}
        >
          <h3 style={{ margin: '0 0 0.25rem 0' }}>{challenge.title}</h3>
          <p style={{ margin: 0 }}>
            Difficulty: {challenge.difficulty} | Tags: {challenge.tags.join(', ')}
          </p>
        </div>
      ))}
    </div>
  );
}
