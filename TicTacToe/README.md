# TicTacToe
TicTacToe - Tic Tac Toe Reinforcement Learning with PyTorch

This project demonstrates a simple reinforcement learning approach to train an agent to play Tic Tac Toe.

## Objectives

The objective of this project is to teach the agent how to play Tic Tac Toe optimally through self-play and experience-based learning. The goal is for the agent to learn:
- Basic reinforcement learning concepts.
- Policy gradient methods.
- Q-learning.
- Self-play and experience-based learning.
- Model-based reinforcement learning.

## Hyperparameters

The hyperparameters used in the training process include:
- Epochs: 16000 (number of training iterations)
- Gamma: 0.95 (discount factor for future rewards)
- Epsilon: 1.0 (exploration rate) [Decay over time]
- MinEpsilon: 0 (minimum exploration rate)
- Epsilon Decay: 0.9998 (rate at which epsilon decreases)
- Learning Rate: 1e-3 (learning rate for weight updates)


# How to use

1. You can start train a new model by running the following command:
   ```bash
   python train.py
   ```

2. After training, you can play against the trained model by running:
   ```bash
   python index.py
   ```


