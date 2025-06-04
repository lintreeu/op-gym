import { Link } from 'react-router-dom';
import { FaGithub } from 'react-icons/fa';

function GitHubCorner() {
  return (
    <a
      href="https://github.com/lintreeu/op-gym"
      target="_blank"
      rel="noopener noreferrer"
      style={{
        position: 'fixed',
        top: '16px',
        right: '16px',
        zIndex: 1000,
        backgroundColor: '#fff',
        border: '1px solid #ddd',
        borderRadius: '999px',
        padding: '8px',
        boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer'
      }}
      title="View on GitHub"
    >
      <FaGithub size={20} color="#000" />
    </a>
  );
}

export default function HomePage() {
  return (
    <div style={{ padding: '40px' }}>
      <h1>Welcome to Op-Gym</h1>
      <p>Learn how GPU operators access memory visually.</p>

      <GitHubCorner />
      <p>Choose an option to get started:</p>
      <ul className="list-disc ml-6">
        <li>
          <Link to="/playground" className="text-blue-600 underline">
            Playground
          </Link>
        </li>
        <li>
          <Link to="/challenges" className="text-blue-600 underline">
            Challenges
          </Link>
        </li>
      </ul>
    </div>
  );
}
