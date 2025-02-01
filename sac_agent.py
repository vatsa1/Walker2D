import torch
import torch.nn as nn
import torch.optim as optim
from models import Actor, Critic 

class SACAgent:
    def __init__(self, state_dim, action_dim, min_action, max_action, device, alpha=0.2, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.soft_update(self.critic_target, self.critic, 1.0)

        self.min_action = min_action
        self.max_action = max_action
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

    def predict(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size, writer, step):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(1 - done).to(self.device)

        # Train Critic
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (done * self.gamma * target_Q)

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        self.soft_update(self.critic_target, self.critic, self.tau)

        # Write losses to TensorBoard
        writer.add_scalar("Critic Loss", critic_loss.item(), step)
        writer.add_scalar("Actor Loss", actor_loss.item(), step)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
