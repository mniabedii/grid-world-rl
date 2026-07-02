# GridWorld Reinforcement Learning

This project implements and compares **Tabular Q-Learning** & **Deep Q-Networks (DQN)** reinforcement learning algorithms on a custom GridWorld environment.

## The Environment

The custom environment is a **5×5 GridWorld** with deterministic actions and static obstacles. The agent's objective is to navigate from the start position to the goal state while maximizing cumulative rewards.

### Grid Layout

```text
S . . . .
. X . X .
. X . . .
. . X . .
. . . . G
```

- **S**: Start position `(0,0)`
- **G**: Goal state `(4,4)`
- **X**: Wall / Obstacle
- **.**: Free space

### Action Space

Discrete actions:

- Up
- Down
- Left
- Right

### Reward Structure

| Event | Reward |
|--------|-------:|
| Standard step | **-1** |
| Move out of bounds | **-5** |
| Hit a wall | **-10** |
| Reach the goal | **+100** |

The reward design encourages the agent to find the shortest valid path while avoiding invalid moves.

---

## 1. Tabular Q-Learning

A classical reinforcement learning algorithm designed for environments with small, discrete state spaces.

The Q-values are updated using the Bellman equation:

<p align="center">

$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]$

</p>

The agent consistently learns the optimal shortest-path policy, requiring fewer steps as training progresses.

---

## 2. Deep Q-Network (DQN)

Instead of storing values in a Q-table, DQN approximates the action-value function using a **Multi-Layer Perceptron (MLP)** implemented with **PyTorch**.

### Features

- Experience Replay Buffer
- Target Network
- ε-greedy exploration
- Adam optimizer
- Mean Squared Error (MSE) loss

Target value:

<p align="center">

$y=r+\gamma\max_{a'}Q_{\text{target}}(s',a')$

</p>

The current focus of this project is understanding why the DQN agent struggles despite the simplicity of the environment.

---

## Experiments & Project Report

After successfully implementing the Tabular Q-Learning agent, the project focused on developing and improving the Deep Q-Network (DQN) agent. As training challenges emerged, multiple experiments and iterative modifications were carried out to investigate the causes of the observed behavior and improve performance.

A complete record of these experiments—including the motivation behind each change, implementation details, results, and analysis—is documented in **`report/analysis.ipynb`**.

The notebook serves as the primary documentation for the project and includes:

- Performance plots and learning curves
- Analysis of the learned policies
- Discussion of the DQN convergence issues
- Experiments, observations, and proposed improvements

---

## Prerequisites & How To Run

- Python 3.12+

Install dependencies:

```bash
pip install -r requirements.txt
```
## Running the Agents

### The Q-Learning Agent

```bash
python3 -m src.qlearning.train_qlearning
```

### The DQN Agent

```bash
python3 -m src.dqn.train_dqn
```
---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── outputs/
│   ├── dqn_steps_per_episode.png
│   └── qlearning_steps_per_episode.png
├── report/
│   └── analysis.ipynb
└── src/
    ├── gridworld.py
    ├── util.py
    ├── qlearning/
    │   ├── qlearning_agent.py
    │   └── train_qlearning.py
    └── dqn/
        ├── dqn_agent.py
        ├── network.py
        ├── replay_buffer.py
        └── train_dqn.py
```

## Technologies Used

- Python
- NumPy
- PyTorch
- Matplotlib
- Jupyter Notebook

---