'use client';

 // @ts-ignore
 // @ts-ignore
import * as ort from 'onnxruntime-web';
import Board from './Board';
import { useState, useEffect } from 'react';

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

const directions = [
  [0, 1], [0, -1], [1, 0], [-1, 0], // Horizontal and vertical
  [1, 1], [1, -1], [-1, 1], [-1, -1]  // Diagonal
];

const isValidMove = (board: string[][], player: string, row: number, col: number): boolean => {
  if (board[row][col] !== '') {
    return false;
  }

  for (const [dr, dc] of directions) {
    let currentRow = row + dr;
    let currentCol = col + dc;
    let hasOpponentPiece = false;

    while (
      currentRow >= 0 && currentRow < 8 &&
      currentCol >= 0 && currentCol < 8 &&
      board[currentRow][currentCol] !== '' &&
      board[currentRow][currentCol] !== player
    ) {
      hasOpponentPiece = true;
      currentRow += dr;
      currentCol += dc;
    }

    if (
      hasOpponentPiece &&
      currentRow >= 0 && currentRow < 8 &&
      currentCol >= 0 && currentCol < 8 &&
      board[currentRow][currentCol] === player
    ) {
      return true;
    }
  }

  return false;
};

const hasValidMoves = (board: string[][], player: string): boolean => {
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      if (isValidMove(board, player, row, col)) {
        return true;
      }
    }
  }
  return false;
};

export default function Page() {
  const [board, setBoard] = useState(initialBoard);
  const [currentPlayer, setCurrentPlayer] = useState('W');
  const [gameOver, setGameOver] = useState(false);

  const handleClick = (row: number, col: number) => {
    if (gameOver || board[row][col] !== '') {
      return;
    }

    if (!isValidMove(board, currentPlayer, row, col)) {
      return;
    }

    const newBoard = board.map(rowArray => [...rowArray]); // Create a deep copy
    newBoard[row][col] = currentPlayer;

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

      const nextPlayer = currentPlayer === 'B' ? 'W' : 'B';
      if (!hasValidMoves(newBoard, nextPlayer)) {
        if (!hasValidMoves(newBoard, currentPlayer)) {
          setGameOver(true);
          return;
        } else {
          // Skip the next player's turn
          setCurrentPlayer(nextPlayer);
          return;
        }
      }
      setCurrentPlayer(nextPlayer);
    }
  };

  const countPieces = (board: string[][], player: string): number => {
    let count = 0;
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        if (board[row][col] === player) {
          count++;
        }
      }
    }
    return count;
  };

  const getWinner = (): string => {
    const blackCount = countPieces(board, 'B');
    const whiteCount = countPieces(board, 'W');

    if (blackCount > whiteCount) {
      return 'Black';
    } else if (whiteCount > blackCount) {
      return 'White';
    } else {
      return 'Tie';
    }
  };

  const resetGame = () => {
    setBoard(initialBoard);
    setCurrentPlayer('W');
    setGameOver(false);
  };

  const skipTurn = () => {
    if (!hasValidMoves(board, currentPlayer)) {
      const nextPlayer = currentPlayer === 'B' ? 'W' : 'B';
      setCurrentPlayer(nextPlayer);
    }
  };

function boardToNumeric(board: string[][]): number[] {
  // Convert board representation to a flat numeric array:
  // 'B' -> 1, 'W' -> -1, empty ('') -> 0
  const numeric: number[] = [];
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[i].length; j++) {
      const cell = board[i][j];
      if (cell === 'B') numeric.push(1);
      else if (cell === 'W') numeric.push(-1);
      else numeric.push(0);
    }
  }
  return numeric;
}

const aiMove = async () => {
  // Compute valid moves for AI ('B')
  const validMoves: number[] = [];
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      if (isValidMove(board, 'B', i, j)) {
        validMoves.push(i * 8 + j);
      }
    }
  }
  if (validMoves.length === 0) {
    setCurrentPlayer('W');
    return;
  }

  try {
    const session = await ort.InferenceSession.create('/a3c_model.onnx');
    const numericBoard = boardToNumeric(board);
    const tensor = new ort.Tensor('float32', Float32Array.from(numericBoard), [1, 64]);
    const feeds: Record<string, ort.Tensor> = { 'input': tensor };
    const results = await session.run(feeds);
    const policy = results['policy'].data as Float32Array;
    // Select the best move among valid moves.
    let bestIndex = validMoves[0];
    let bestProb = policy[bestIndex];
    for (const idx of validMoves) {
      if (policy[idx] > bestProb) {
        bestProb = policy[idx];
        bestIndex = idx;
      }
    }
    const row = Math.floor(bestIndex / 8);
    const col = bestIndex % 8;
    console.log(`AI (ONNX) selected move at row ${row}, col ${col} with probability ${bestProb}`);
    handleClick(row, col);
  } catch (error) {
    console.error('Error during AI move using ONNX model:', error);
  }
};

  useEffect(() => {
    if (currentPlayer === 'B' && !gameOver) {
      const timer = setTimeout(() => {
        aiMove();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [currentPlayer, board, gameOver]);

  return (
    <div>
      <h1>Othello</h1>
      <p>Current Player: {currentPlayer === 'B' ? 'Black' : 'White'}</p>
      {gameOver && (
        <p>
          Game Over! Winner: {getWinner()}
        </p>
      )}
      <Board board={board} onClick={handleClick} />
      <button onClick={resetGame}>Reset Game</button>
      <button onClick={skipTurn}>Skip Turn</button>
    </div>
  );
}
