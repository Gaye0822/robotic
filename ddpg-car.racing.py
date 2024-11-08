import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# CNN yapýsý (Car Racing ortamý için)
class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CNNActorCritic, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Flatten the output for the fully connected layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers for actor (policy) and critic (Q-function)
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3_actor = nn.Linear(128, action_space.shape[0])
        self.fc3_critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3_actor(x), self.fc3_critic(x)

    def act(self, obs):
        action, _ = self.forward(obs)
        return action

# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, env, actor_critic=CNNActorCritic, gamma=0.99, tau=0.005, lr=1e-3, buffer_size=100000):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ac = actor_critic(env.observation_space, env.action_space).to(self.device)
        self.actor_optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        
        self.target_ac = actor_critic(env.observation_space, env.action_space).to(self.device)
        self.update_target()

    def update_target(self):
        for target_param, param in zip(self.target_ac.parameters(), self.ac.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
        action = self.ac.act(obs).cpu().detach().numpy()[0]
        return action

    def update(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Critic loss (TD error)
        next_action = self.ac.act(next_states)
        next_q_value = self.target_ac.q(next_states, next_action)
        target_q_value = rewards + self.gamma * (1 - dones) * next_q_value
        
        current_q_value = self.ac.q(states, actions)
        critic_loss = (current_q_value - target_q_value).pow(2).mean()
        
        # Actor loss (policy gradient)
        actor_loss = -self.ac.q(states, self.ac.act(states)).mean()
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Target update (soft update)
        self.update_target()

    def save(self, filepath):
        torch.save(self.ac.state_dict(), filepath)

    def load(self, filepath):
        self.ac.load_state_dict(torch.load(filepath))

# Main training loop for Car Racing environment
def train():
    env = gym.make("CarRacing-v2")
    agent = DDPGAgent(env)
    
    num_episodes = 100
    max_timesteps = 1000
    batch_size = 64
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        
        for t in range(max_timesteps):
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.update(batch_size)
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")
    
    agent.save("ddpg_carracing_model")
    env.close()

train()
