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
    if (board[row][col] !== '') {
      return;
    }

    const newBoard = board.map(rowArray => [...rowArray]); // Create a deep copy
    newBoard[row][col] = currentPlayer;

    const directions = [
      [0, 1], [0, -1], [1, 0], [-1, 0], // Horizontal and vertical
      [1, 1], [1, -1], [-1, 1], [-1, -1]  // Diagonal
    ];

    let piecesToFlip: [number, number][] = [];

    for (const [dr, dc] of directions) {
      let currentRow = row + dr;
      let currentCol = col + dc;
      let piecesInDirection: [number, number][] = [];

      while (
        currentRow >= 0 && currentRow < 8 &&
        currentCol >= 0 && currentCol < 8 &&
        newBoard[currentRow][currentCol] !== '' &&
        newBoard[currentRow][currentCol] !== currentPlayer
      ) {
        piecesInDirection.push([currentRow, currentCol]);
        currentRow += dr;
        currentCol += dc;
      }

      if (
        currentRow >= 0 && currentRow < 8 &&
        currentCol >= 0 && currentCol < 8 &&
        newBoard[currentRow][currentCol] === currentPlayer
      ) {
        piecesToFlip = piecesToFlip.concat(piecesInDirection);
      }
    }

    if (piecesToFlip.length > 0) {
      piecesToFlip.forEach(([r, c]) => {
        newBoard[r][c] = currentPlayer;
      });

      setBoard(newBoard);
      setCurrentPlayer(currentPlayer === 'B' ? 'W' : 'B');
    }
  };

  return (
    <div>
      <h1>Othello</h1>
      <Board board={board} onClick={handleClick} />
    </div>
  );
}
