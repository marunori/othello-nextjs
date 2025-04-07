#!/usr/bin/env python3
"""
A3C Agent Implementation using PyTorch for Othello Reinforcement Learning

This implementation provides a basic example of an Asynchronous Advantage Actor-Critic (A3C) agent.
It defines an Actor-Critic network, a training process using multiprocessing, and a match between two AI players.
In the match, the winner receives a reward of 1.0 and the global network is updated using the winning transitions.
Note: This example uses dummy state inputs and rewards as placeholders.
Replace the dummy data with actual game state representations and reward signals from the Othello environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from othello_env import OthelloEnv

# Actor-Critic Network definition
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_space):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_space)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc(x))
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value

# A3C Agent class
class A3CAgent:
    def __init__(self, input_dim=64, hidden_dim=128, action_space=64, lr=1e-4):
        self.global_network = ActorCritic(input_dim, hidden_dim, action_space)
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=lr)
        self.global_network.share_memory()  # Enable sharing for multiprocessing
        manager = mp.Manager()
        self.replay_buffer = manager.list()
        
    def train(self, num_processes=4, num_steps=20):
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=self.train_process, args=(rank, num_steps))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    def train_process(self, rank, num_steps):
        # Create local network and synchronize with the global network
        local_network = ActorCritic(64, 128, 64)
        local_network.load_state_dict(self.global_network.state_dict())
        optimizer = optim.Adam(local_network.parameters(), lr=1e-4)
        
        for step in range(num_steps):
            # Replace this with an actual state representation from the Othello environment
            state = torch.FloatTensor(np.random.rand(64))
            policy, value = local_network(state)
            
            # Dummy reward and simple loss calculation for demonstration purposes
            reward = torch.tensor(np.random.rand(), dtype=torch.float32)
            advantage = reward - value
            
            actor_loss = -torch.log(policy.max()) * advantage
            critic_loss = advantage.pow(2)
            loss = actor_loss + critic_loss
            
            # Backpropagation step
            optimizer.zero_grad()
            loss.backward()
            # Push local gradients to the global network
            for global_param, local_param in zip(self.global_network.parameters(), local_network.parameters()):
                if local_param.grad is not None:
                    global_param._grad = local_param.grad
            optimizer.step()
            next_state = torch.FloatTensor(np.random.rand(64))
            transition = (state, int(policy.argmax().item()), reward.item(), next_state, False)
            self.replay_buffer.append(transition)
            print(f"Process {rank} Step {step}: loss {loss.item()}")
            # Synchronize local network with the updated global network
            local_network.load_state_dict(self.global_network.state_dict())
        
        print(f"Process {rank} training completed.")
    
    def match(self):
        """
        Simulate a match between two AI players using the global network.
        Each turn the AI selects the best valid move according to the network's policy.
        At the end, the winner is determined by stone count. If there is a winner (no tie),
        the transitions made by the winning player are used to update the global network with a reward of 1.0.
        """
        env = OthelloEnv()
        env.reset()
        transitions = []  # Each element is (state, action_index, player)
        current_player = env.current_player
        while True:
            valid_moves = env.get_valid_moves(current_player)
            if not valid_moves:
                break
            # Prepare state: flatten the board
            state_array = np.array(env.board).flatten()
            state_tensor = torch.FloatTensor(state_array)
            policy, _ = self.global_network(state_tensor)
            # Map valid moves (row, col) to flattened indices
            valid_indices = [r * 8 + c for (r, c) in valid_moves]
            # Select the valid move with highest probability
            probs = policy.detach().numpy()
            best_action_index = max(valid_indices, key=lambda idx: probs[idx])
            action = (best_action_index // 8, best_action_index % 8)
            transitions.append((state_tensor, best_action_index, current_player))
            _, _, done, _ = env.step(action)
            current_player = env.current_player
            if done:
                break
        # Determine winner based on stone counts
        board_np = np.array(env.board)
        count1 = np.sum(board_np == 1)
        count_minus1 = np.sum(board_np == -1)
        if count1 > count_minus1:
            winner = 1
        elif count_minus1 > count1:
            winner = -1
        else:
            winner = 0
        print("Match completed. Winner:", "Player 1" if winner==1 else "Player 2" if winner==-1 else "Tie")
        # Update global network using transitions from the winning player
        if winner != 0:
            for state, action, player in transitions:
                if player == winner:
                    policy, value = self.global_network(state)
                    action_prob = policy[action]
                    actor_loss = -torch.log(action_prob) * 1.0  # reward is 1.0
                    critic_loss = (1.0 - value).pow(2)
                    loss = actor_loss + critic_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            print("Global network updated for winning transitions.")

    def train_from_replay_buffer(self, batch_size, iterations):
        effective_batch_size = min(len(self.replay_buffer), batch_size)
        if effective_batch_size == 0:
            print("No samples in replay buffer")
            return
        for i in range(iterations):
            indices = np.random.choice(len(self.replay_buffer), effective_batch_size, replace=False)
            batch = [self.replay_buffer[idx] for idx in indices]
            loss_total = 0.0
            for state, action, reward, next_state, done in batch:
                policy, value = self.global_network(state)
                action_prob = policy[action]
                actor_loss = -torch.log(action_prob) * reward
                critic_loss = (reward - value).pow(2)
                loss_total += (actor_loss + critic_loss)
            loss_avg = loss_total / effective_batch_size
            self.optimizer.zero_grad()
            loss_avg.backward()
            self.optimizer.step()
            print(f"Replay training iteration {i}: loss {loss_avg.item()}")
    
    def save_model(self, filepath):
        torch.save({
            'global_network_state_dict': self.global_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.global_network.load_state_dict(checkpoint['global_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
    def save_model_onnx(self, filepath):
        try:
            import onnx
        except ImportError:
            print("ONNX module not installed. Please run 'pip install onnx' and try again.")
            return
        dummy_input = torch.randn(1, self.global_network.fc.in_features)
        torch.onnx.export(self.global_network, dummy_input, filepath, input_names=['input'], output_names=['policy', 'value'])
        print(f"Model saved in ONNX format to {filepath}")

    def match_with_onnx(self, onnx_filepath):
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime module not installed. Please run 'pip install onnxruntime' and try again.")
            return
        sess = ort.InferenceSession(onnx_filepath)
        env = OthelloEnv()
        env.reset()
        transitions = []
        current_player = env.current_player
        while True:
            valid_moves = env.get_valid_moves(current_player)
            if not valid_moves:
                break
            state_array = np.array(env.board).flatten().astype(np.float32)
            input_tensor = state_array.reshape(1, -1)
            outputs = sess.run(['policy', 'value'], {'input': input_tensor})
            policy = outputs[0][0]
            valid_indices = [r * 8 + c for (r, c) in valid_moves]
            best_action_index = max(valid_indices, key=lambda idx: policy[idx])
            action = (best_action_index // 8, best_action_index % 8)
            transitions.append((state_array, best_action_index, current_player))
            _, _, done, _ = env.step(action)
            current_player = env.current_player
            if done:
                break
        board_np = np.array(env.board)
        count1 = np.sum(board_np == 1)
        count_minus1 = np.sum(board_np == -1)
        if count1 > count_minus1:
            winner = 1
        elif count_minus1 > count1:
            winner = -1
        else:
            winner = 0
        print("ONNX Match completed. Winner:", "Player 1" if winner==1 else "Player 2" if winner==-1 else "Tie")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    agent = A3CAgent()
    num_cycles = 300  # Set number of cycles for training and matches
    for cycle in range(num_cycles):
        print(f"--- Cycle {cycle+1} ---")
        print("Training agent...")
        agent.train(num_processes=2, num_steps=10)
        print("Starting a match between two AI players...")
        agent.match()
        print("Training global network from replay buffer...")
        agent.train_from_replay_buffer(batch_size=8, iterations=10)
        print("Saving model...")
        agent.save_model("a3c_model.pth")
        print("Saving ONNX model...")
        agent.save_model_onnx("a3c_model.onnx")
    print("Starting a match with ONNX model...")
    agent.match_with_onnx("a3c_model.onnx")
