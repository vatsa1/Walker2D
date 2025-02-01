import gym
import pybullet_envs
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer  # Import from replay_buffer.py
from sac_agent import SACAgent  # Import from sac_agent.py
from utils import running_average

# Hyperparameters
replay_buffer_size = 200000
start_training_after = 5000
train_steps = 1500000
batch_size = 256
exploration_steps = 10000

# Environment setup
env = gym.make("Walker2DBulletEnv-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent and Replay Buffer Initialization
agent = SACAgent(state_dim, action_dim, min_action, max_action, device)
replay_buffer = ReplayBuffer(replay_buffer_size)

# TensorBoard Setup
writer = SummaryWriter()

# Training Loop
state = env.reset()
episode_reward = 0
episode_count = 0
reward_sum = 0
training_rewards = []
actor_losses, critic_losses = [], []
best_reward = 0

for step in range(train_steps):
    # Populate replay buffer with exploration steps
    action = agent.predict(state)
    action = (action + np.random.normal(0, max_action * 0.1, size=action_dim)).clip(min_action, max_action)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.add((state, action, next_state, reward, done))
    state = next_state
    episode_reward += reward
    reward_sum += reward

    # Episode Termination and Logging
    if done:
        state = env.reset()
        episode_count += 1
        if episode_reward > best_reward and episode_reward > 1000:
            model_path = "models"  # Create a folder named "models" to store the models
            os.makedirs(model_path, exist_ok=True)
            torch.save(agent.actor.state_dict(), f"{model_path}/walker2d_sac_actor_step_{step}.pth")
            torch.save(agent.critic.state_dict(), f"{model_path}/walker2d_sac_critic_step_{step}.pth")
            best_reward = episode_reward

        if episode_count % 100 == 0:
            avg_reward_100_episodes = reward_sum / 100
            print(f"Average reward for last 100 episodes (Episode {episode_count}): {avg_reward_100_episodes}")
            reward_sum = 0
        training_rewards.append(episode_reward)
        episode_reward = 0

    # Training the Agent
    if step >= start_training_after and step % 10 == 0:
        actor_loss, critic_loss = agent.train(replay_buffer, batch_size, writer, step)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

# Plotting and visualization
window_size = 100
running_rewards = running_average(training_rewards, window_size)
plt.figure(figsize=(10, 5))
plt.plot(running_rewards)
plt.title("Running Average Reward")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid()
plt.show()

# Plot the training rewards graph
plt.figure()
plt.plot(training_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards per Episode")
plt.show()

# Plot the loss graphs
plt.figure()
plt.plot(actor_losses, label="Actor Loss")
plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Losses during Training")
plt.legend()
plt.show()
