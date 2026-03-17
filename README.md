# Q-Learning Agent for Game Environment

A Reinforcement Learning project implementing a **Q-Learning agent** capable of learning optimal actions within a game environment. The agent interacts with the environment, learns from rewards, and improves its strategy over time using **Q-value updates and exploration policies**.

## About

This project implements a **Q-learning reinforcement learning agent** designed to learn optimal behaviour in a grid-based game environment. 

The agent observes the game state, extracts useful features, evaluates possible actions, and learns from rewards received after each action. Over multiple training episodes, the agent improves its decision-making policy by updating Q-values and balancing exploration and exploitation.

The implementation is built in **Python** and demonstrates key reinforcement learning concepts including:

- Q-value estimation
- Reward-based learning
- Feature extraction from states
- Exploration strategies
- Policy improvement over time

## Technologies Used

- Python
- Reinforcement Learning
- Q-Learning
- Feature-based value approximation
- Game AI environments

## Reinforcement Learning Approach

The agent uses the **Q-Learning algorithm**, a model-free reinforcement learning method that allows an agent to learn optimal policies through interaction with an environment.

At each step the agent:

1. Observes the current state
2. Selects an action
3. Receives a reward
4. Updates its Q-value estimate
5. Moves to the next state

The Q-value update rule is:

Q(s,a) ← (1 − α)Q(s,a) + α[r + γ max Q(s',a')]

Where:

- **α** = learning rate  
- **γ** = discount factor  
- **r** = reward received  
- **s'** = next state  

This allows the agent to gradually learn which actions produce the highest long-term rewards.

## Feature Engineering

The agent extracts useful features from the game state to guide decision making.

Key features include:

- Distance to the closest food
- Whether food will be eaten
- Number of nearby ghosts
- Bias term for baseline learning

These features are combined using a **weighted linear function** to estimate Q-values for state-action pairs.

## Exploration Strategy

The agent uses an **epsilon-greedy exploration strategy** to balance exploration and exploitation.

- With probability ε → choose a random action
- With probability 1 − ε → choose the best known action

This ensures the agent continues exploring new strategies while gradually converging toward an optimal policy.

## Training

The agent is trained over multiple episodes where it interacts with the environment and updates its policy.

Key training parameters include:

- Learning Rate (α): 0.2
- Discount Factor (γ): 0.8
- Exploration Rate (ε): 0.05
- Training Episodes: configurable

During training the agent continuously updates feature weights based on rewards received from the environment.

## Project Structure
project/
│
├── game.py                # Game engine and environment logic
├── pacmanAgents.py       # Baseline Pacman agents
├── ghostAgents.py        # Ghost behaviour agents
├── keyboardAgents.py     # Human-controlled agent via keyboard
├── mlLearningAgents.py   # Q-Learning agent implementation
├── graphicsDisplay.py    # Graphical display system
├── graphicsUtils.py      # Rendering utilities
├── textDisplay.py        # Console-based display
├── layout.py             # Game map layout handling
├── util.py               # Data structures and helper utilities
└── projectParams.py      # Project configuration

## Example Run

Run the game environment and agent:

```bash
python pacman.py
```


---

# Learning Outcomes

This project demonstrates:

- Implementation of a reinforcement learning agent
- Feature-based Q-value estimation
- Exploration vs exploitation strategies
- Reward-based policy learning
- AI decision-making in dynamic environments

## License

This project builds upon the Pacman AI projects developed at UC Berkeley for educational purposes. Please retain attribution when using or extending this work.
