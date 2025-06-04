import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div className="p-6 space-y-4">
      <h1 className="text-3xl font-bold">Welcome to Op-Gym</h1>
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
