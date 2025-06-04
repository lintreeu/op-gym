import { Routes, Route} from 'react-router-dom';
import HomePage from './pages/HomePage';
import PlaygroundPage from './pages/PlaygroundPage';
import ChallengesPage from './pages/ChallengesPage';
import ChallengeDynamicPage from './pages/ChallengeDynamicPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/playground" element={<PlaygroundPage />} />
      <Route path="/challenges" element={<ChallengesPage />} />
      <Route path="/challenges/:challengeId" element={<ChallengeDynamicPage />} />
    </Routes>
  );
}
