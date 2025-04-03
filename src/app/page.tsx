'use client';

import Board from './Board';
import { useState } from 'react';

const initialBoard = [
  ['', '', '', '', '', '', '', ''],
  ['', '', '', '', '', '', '', ''],
  ['', '', '', '', '', '', '', ''],
  ['', '', '', 'W', 'B', '', '', ''],
  ['', '', '', 'B', 'W', '', '', ''],
  ['', '', '', '', '', '', '', ''],
  ['', '', '', '', '', '', '', ''],
  ['', '', '', '', '', '', '', ''],
];

export default function Page() {
  const [board, setBoard] = useState(initialBoard);
  const [currentPlayer, setCurrentPlayer] = useState('B');

  const handleClick = (row: number, col: number) => {
    // Check if the cell is empty
    if (board[row][col] !== '') {
      return;
    }

    // Update the board state
    const newBoard = board.map((rowArray, rowIndex) => {
      return rowArray.map((cell, colIndex) => {
        if (rowIndex === row && colIndex === col) {
          return currentPlayer;
        }
        return cell;
      });
    });

    // Update the state
    setBoard(newBoard);
    setCurrentPlayer(currentPlayer === 'B' ? 'W' : 'B');
  };

  return (
    <div>
      <h1>Othello</h1>
      <Board board={board} onClick={handleClick} />
    </div>
  );
}
