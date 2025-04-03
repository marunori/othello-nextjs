'use client';

import React from 'react';

interface BoardProps {
  board: string[][];
  onClick: (row: number, col: number) => void;
}

const Board: React.FC<BoardProps> = ({ board, onClick }) => {
  return (
    <div>
      {board.map((row, rowIndex) => (
        <div key={rowIndex} style={{ display: 'flex' }}>
          {row.map((cell, colIndex) => (
            <button
              key={colIndex}
              onClick={() => onClick(rowIndex, colIndex)}
              style={{
                width: '50px',
                height: '50px',
                backgroundColor: 'green',
                border: '2px solid black',
                position: 'relative',
              }}
            >
              {cell === 'B' && <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'black' }}></div>}
              {cell === 'W' && <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'white' }}></div>}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
};

export default Board;
