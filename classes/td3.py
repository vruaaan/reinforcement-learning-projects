import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .agents import Agent
    from .buffers import ReplayBuffer
except ImportError:
    from agents import Agent
    from buffers import ReplayBuffer


class TD3Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


class TD3Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TD3Agent(Agent):
    def __init__(
        self,
        env=None,
        name: str = "bot",
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        expl_noise=0.1,
        batch_size=256,
        buffer_capacity=100_000,
        lr=1e-3,
    ):
        super().__init__(env=env, name=name)
        self.actor = TD3Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic1 = TD3Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = TD3Critic(self.obs_dim, self.action_dim).to(self.device)
        self.tgt_actor = copy.deepcopy(self.actor)
        self.tgt_critic1 = copy.deepcopy(self.critic1)
        self.tgt_critic2 = copy.deepcopy(self.critic2)

        for net in [self.tgt_actor, self.tgt_critic1, self.tgt_critic2]:
            for param in net.parameters():
                param.requires_grad = False

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.total_updates = 0

    def act(self, obs, deterministic=False):
        obs_t = self.obs_tensor(obs)
        self.actor.eval()
        with torch.no_grad():
            raw = self.actor(obs_t)
        self.actor.train()

        action = self.rescale(raw).cpu().numpy()[0]
        if not deterministic:
            noise = np.random.normal(
                0,
                self.expl_noise * (self.action_high_np - self.action_low_np),
                size=action.shape,
            )
            action = action + noise
        return self.clip_action(action)

    def td3_update(self):
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            raw_next = self.tgt_actor(next_obs)
            next_actions = self.rescale(raw_next)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = torch.max(
                torch.min(next_actions + noise, self.action_high),
                self.action_low,
            )
            q1_next = self.tgt_critic1(next_obs, next_actions)
            q2_next = self.tgt_critic2(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1.0 - dones) * q_next

        q1_current = self.critic1(obs, actions)
        q2_current = self.critic2(obs, actions)
        critic1_loss = nn.functional.mse_loss(q1_current, target_q)
        critic2_loss = nn.functional.mse_loss(q2_current, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            raw_actions = self.actor(obs)
            actor_actions = self.rescale(raw_actions)
            actor_loss = -self.critic1(obs, actor_actions).mean()

            self.critic1_opt.zero_grad()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for main, target in [
                (self.actor, self.tgt_actor),
                (self.critic1, self.tgt_critic1),
                (self.critic2, self.tgt_critic2),
            ]:
                for p_main, p_tgt in zip(main.parameters(), target.parameters()):
                    p_tgt.data.mul_(1.0 - self.tau)
                    p_tgt.data.add_(self.tau * p_main.data)

    def train(self, total_timesteps=200_000, learning_starts=1_000, log_every=20, render=True):
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_count = 0

        for timestep in range(1, total_timesteps + 1):
            if timestep < learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, action, reward, next_obs, float(terminated))
            episode_reward += reward
            obs = next_obs

            if done:
                episode_count += 1
                if episode_count % log_every == 0 and render:
                    print(
                        f"Timestep {timestep:>7} | Episode {episode_count:>4} | "
                        f"Reward: {episode_reward:>8.2f}"
                    )
                episode_reward = 0.0
                obs, _ = self.env.reset()

            if timestep >= learning_starts and len(self.replay_buffer) >= self.batch_size:
                self.td3_update()

        print("Training complete.")
        return self
