# Snake Game with Deep Q-Learning

## Overview
This project is an innovative version of the classic Snake game, enhanced with deep Q-learning, a form of reinforcement learning. Developed in Python, it integrates Pygame for game rendering and TensorFlow for implementing the reinforcement learning model.

## Table of Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [Technologies Used](#technologies-used)
- [Running the Project](#running-the-project)
- [Snake Perception and Decision-Making](#snake-perception-and-decision-making)
- [Learning process](#learning-process)
- [Model Performance and Effectiveness](#model-performance-and-effectiveness)

## Key Components
- **train_snake.py**: Handles the training of the snake agent using deep Q-learning.
- **Snake_presentation.py**: Visualizes the performance of the trained model.
- **game_agent.py**: Contains the deep Q-learning model, training logic, and buffer management.
- **pygame_client.py**: Manages the game's display and user interactions.
- **snake_game.py**: Defines the game's mechanics like snake movements and reward system.

## Technologies Used
- **Pygame**: For game rendering and interactions.
- **TensorFlow**: For the deep Q-learning model.

## Running the Project 🚀
1. Train the Agent: Run `python train_snake.py`.
2. Visualize the Agent: Execute `python Snake_presentation.py`.



![output-onlinegiftools](https://github.com/p3choco/Reinforced_Python_Game/assets/62072811/bc9d105d-4e99-41df-8b97-2808a9cfae94)


# How It Works?



## Snake Perception and Decision-Making 🐍
1. **Sensory Awareness**: The snake is programmed to perceive its immediate surroundings. It can detect obstacles within a one-block radius around its head.
2. **Directional Awareness**: The snake is aware of its current movement direction, which plays a crucial role in its decision-making process.
3. **Food Detection**: In addition to sensing its immediate surroundings, the snake is also aware of the direction in which food is located.

## Learning process
1. **Initial Snake Behavior**: Initially, the snake's movements are largely random.
2. **Game State Buffering**: As the game progresses, states from each game are stored in a buffer.
3. **Training Process**:
   - The buffer retains information like the current state, the next state, and the reward associated with each state.
   - Rewards for future states are adjusted using a discount factor.
   - The deep neural network aims to predict optimal moves for upcoming states.
   - Target values for training are derived using Bellman's equation.
   - Gradients are generated based on the difference between predicted and actual outcomes.
   - The model updates its weights according to these gradients.
4. **Reinforcement Learning Epochs**: Over numerous epochs, the snake increasingly relies on the deep network for decision-making.
5. **Model Preservation**: Once adequately trained, the model is saved for future use.

## Model Performance and Effectiveness
1. **Training Epochs**: The model reached its peak performance after approximately 600 training epochs.
2. **Average Score Achievement**: In its final form, the model consistently maintains an average score of around 30 points.
3. **Significance of the Score**: Considering the dimensions of the game board (20x20), achieving a 30-point score is a strong indicator of the model's effectiveness and strategic gameplay.



