import React, { useState } from 'react';

const AiMoveFetcher: React.FC = () => {
  const [move, setMove] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const getNextMove = async () => {
    setLoading(true);
    try {
      // Pass game state value if required, here "some_state" is a placeholder.
      const response = await fetch('/api/ai-move?state=some_state');
      const data = await response.json();
      setMove(data.move);
    } catch (err) {
      console.error('Error fetching AI move:', err);
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>AI Next Move</h2>
      <button onClick={getNextMove} disabled={loading}>
        {loading ? 'Loading...' : 'Get Next Move'}
      </button>
      {move && <p>Move: {move}</p>}
    </div>
  );
};

export default AiMoveFetcher;
