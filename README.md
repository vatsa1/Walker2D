# Walker2D Reinforcement Learning Project

## Overview
This project presents a comparative analysis of two reinforcement learning (RL) algorithms—**Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)**—on the Walker2D environment. The aim is to evaluate their effectiveness in training a robotic agent to walk efficiently in a simulated environment.

## Motivation
Traditional control methods struggle with dynamic and complex environments, making it difficult to predefine control strategies for every possible scenario. Reinforcement learning provides a solution by allowing autonomous learning through interaction with the environment, making it particularly useful for robotic applications.

## Algorithms Implemented
### 1. Soft Actor-Critic (SAC)
SAC is an off-policy RL algorithm that integrates entropy regularization, improving exploration and sample efficiency. It is particularly suited for continuous control tasks.
#### Key Features:
- High sample efficiency
- Encourages exploration via entropy maximization
- Well-suited for robotic control and autonomous driving applications

### 2. Proximal Policy Optimization (PPO)
PPO is an on-policy RL algorithm that ensures stable training through trust region policy optimization.
#### Key Features:
- Stable training with clipped policy updates
- High performance in continuous action spaces
- Frequently used in simulated environments such as OpenAI Gym and MuJoCo

## Project Goals
- Compare the performance of SAC and PPO in the **Walker2D** environment.
- Evaluate training efficiency, stability, and robustness.
- Identify strengths and weaknesses of each algorithm in a continuous control setting.

## Implementation Details
- **Environment**: MuJoCo-based Walker2D from OpenAI Gym
- **Frameworks Used**:
  - TensorFlow/PyTorch (for deep reinforcement learning)
  - OpenAI Gym (for RL environments)
  - MuJoCo (for physics-based simulation)
- **Training Process**:
  - SAC trained for 1.5 million timesteps (training time: ~3476s)
  - PPO trained for 1 million timesteps (training time: ~5787.92s)

## Results
- **SAC** demonstrated higher sample efficiency and faster convergence.
- **PPO** produced more stable results but was slower due to its on-policy nature.
- SAC's off-policy learning and soft value function updates made it more efficient for continuous control tasks.
- PPO was easier to implement and tune, making it a preferred choice for stable training.

### Performance Plots
- [PPO Running Reward Plot](http://drive.google.com/file/d/1wAoXqdXNA3QG4f3dIF4SoCSaqw2_lz4S/view)
- SAC vs PPO simulation results

## Conclusion
Both algorithms have their advantages:
- **SAC** is more sample-efficient and achieves better performance but is complex to tune.
- **PPO** is more stable but slower and less sample-efficient.

## Authors
- **Shrivatsa Mudligiri**
- **Rufina George**

## References
- OpenAI Gym: https://gym.openai.com/
- MuJoCo: http://www.mujoco.org/
- SAC Paper: https://arxiv.org/abs/1801.01290
- PPO Paper: https://arxiv.org/abs/1707.06347

