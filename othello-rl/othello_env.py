#!/usr/bin/env python3
"""
Othello Environment for Reinforcement Learning

This basic implementation provides an environment for Othello (Reversi)
games. It follows a structure similar to OpenAI Gym environments, with
methods such as reset(), step(), and render(). Note: Valid move calculation
and stone flipping rules are not fully implemented. This is a starting
point for developing a reinforcement learning based Othello AI.
"""

class OthelloEnv:
    def __init__(self):
        self.size = 8
        self.reset()

    def reset(self):
        """Reset the board to initial state."""
        # Initialize an 8x8 board with zeros.
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        mid = self.size // 2
        # Set up the initial four stones.
        self.board[mid - 1][mid - 1] = 1
        self.board[mid][mid] = 1
        self.board[mid - 1][mid] = -1
        self.board[mid][mid - 1] = -1
        # Set starting player: 1 for X, -1 for O.
        self.current_player = 1
        return self.board

    def render(self):
        """Print the current board state."""
        symbol = {0: '.', 1: 'W', -1: 'B'}
        for row in self.board:
            print(' '.join(symbol[cell] for cell in row))
        print()

    def get_valid_moves(self, player):
        """
        Return a list of valid moves for the given player.
        NOTE: This function currently serves as a placeholder.
        A complete implementation should calculate legal moves based on 
        Othello rules.
        """
        valid_moves = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        return valid_moves

    def step(self, action):
        """
        Execute an action (a tuple (row, col)) and update the board.
        Returns: observation, reward, done, info.
        """
        # For now, check if action is in valid moves.
        valid_moves = self.get_valid_moves(self.current_player)
        if action not in valid_moves:
            # Invalid move yields a negative reward and terminates the episode.
            return self.board, -1, True, {"info": "Invalid move."}

        # Dummy placement without stone flipping.
        row, col = action
        self.board[row][col] = self.current_player

        # Reward and done status are placeholders.
        reward = 0
        done = False

        # Swap player turn.
        self.current_player *= -1
        return self.board, reward, done, {}

if __name__ == '__main__':
    env = OthelloEnv()
    print("Initial board state:")
    env.reset()
    env.render()
