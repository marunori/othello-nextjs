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
                backgroundColor: cell === 'B' ? 'black' : cell === 'W' ? 'white' : 'lightgreen',
                borderRadius: '50%',
                border: '1px solid gray',
              }}
            >
            </button>
          ))}
        </div>
      ))}
    </div>
  );
};

export default Board;
