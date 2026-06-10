# Q-Learning Agent in Simple GridWorld

A basic implementation of a Q-learning agent in a custom GridWorld environment using NumPy.
This project demonstrates core reinforcement learning concepts in a simple and readable way.

## Overview

The agent learns to navigate a 5×5 GridWorld from a start position to a goal while avoiding walls. The environment is fully implemented from scratch without using RL libraries like Gym.

## Environment
```
S . . . .
. X . X .
. X . . .
. . X . .
. . . . G
```

- `S`: Start position  
- `G`: Goal state  
- `X`: Wall (blocked cell with punishment)  
- `.`: Free space  

## Algorithm

The agent uses **Q-Learning**.
```
Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]
```
Where:
- `α` = learning rate  
- `γ` = discount factor  
- `ε` = exploration rate (ε-greedy policy)

## How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/qlearning-agent-simple-grid-world.git
cd qlearning-agent-simple-grid-world
```

### 2. Install dependencies
```
pip install numpy matplotlib
```

### 3. Run
```
python gridworld.py
```

## Output
After training, the agent learns an optimal policy such as:
```
S → → → ↓
↑ X ↓ X ↓
↑ X → → ↓
→ → X → ↓
→ → → → G
```
A learning curve is also displayed showing improvement over episodes.

## This Project Demonstrates:
- Markov Decision Process (MDP) basics
- Q-learning update rule
- Exploration vs exploitation