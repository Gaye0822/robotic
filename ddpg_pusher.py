import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from core import MLPActorCritic  # DDPG i�in uygun bir model
from replay_buffer import ReplayBuffer  # Replay buffer import edilmeli

class DDPGAgent:
    def __init__(self, env, actor_critic=MLPActorCritic, gamma=0.99, tau=0.005, lr=1e-3):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-Critic modeli (DDPG'de akt�r deterministik bir politikaya dayan�r)
        self.ac = actor_critic(env.observation_space, env.action_space).to(self.device)
        self.actor_optimizer = optim.Adam(self.ac.pi.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.ac.q_network.parameters(), lr=lr)

        # Hedef a�
        self.target_ac = actor_critic(env.observation_space, env.action_space).to(self.device)
        self.update_target()

        # Replay buffer
        self.replay_buffer = None  # Replay buffer ba�lang��ta None, d��ar�dan ayarlanacak

    def set_replay_buffer(self, replay_buffer):
        # D��ar�dan verilen replay buffer'� agent i�ine ata
        self.replay_buffer = replay_buffer

    def update_target(self):
        # Hedef a�� g�ncelleme (soft update)
        for target_param, param in zip(self.target_ac.parameters(), self.ac.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, obs):
        # G�zlemi torch tens�re �evir ve eylemi al
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        action = self.ac.pi(obs)  # Akt�r a��ndan deterministik bir eylem al
        return action.detach().cpu().numpy()  # detach ile grafikten ay�r

    def update(self, batch_size=64):
        # Replay bufferdan �rnekleme yap
        if self.replay_buffer is None:
            raise ValueError("Replay buffer not set. Use set_replay_buffer to assign a buffer.")
        
        # Replay buffer'dan bir batch �rne�i al
        states, actions, rewards, next_states, not_dones = self.replay_buffer.sample(batch_size)
        
        # Tens�rlere �evir ve cihaza g�nder
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        not_dones = torch.as_tensor(not_dones, dtype=torch.float32).to(self.device)

        # Q hedef de�erini hesapla (DDPG'de sadece Q ��renme)
        with torch.no_grad():
            next_action = self.target_ac.pi(next_states)  # Hedef akt�r a�� kullanarak eylemi al
            target_q = rewards + self.gamma * not_dones * self.target_ac.q_network(next_states, next_action)

        # Kritik a� kayb�
        q = self.ac.q_network(states, actions)
        critic_loss = ((q - target_q) ** 2).mean()

        # Kritik g�ncellemesi
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Akt�r a�� kayb� (DDPG'de akt�r, Q de�erini maksimize etmeye �al���r)
        actor_loss = -self.ac.q_network(states, self.ac.pi(states)).mean()

        # Akt�r g�ncellemesi
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Hedef a� g�ncellemesi (soft update)
        self.update_target()

    def save(self, filepath):
        # Modeli dosyaya kaydet
        torch.save(self.ac.state_dict(), filepath)

    def load(self, filepath):
        # Modeli dosyadan y�kle
        self.ac.load_state_dict(torch.load(filepath))
