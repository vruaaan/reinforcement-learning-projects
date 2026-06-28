import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

try:
    from .agents import Agent
    from .buffers import ReplayBuffer
except ImportError:
    from agents import Agent
    from buffers import ReplayBuffer


class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_NN = nn.Linear(256, action_dim)
        self.log_std_NN = nn.Linear(256, action_dim)

    def forward(self, obs):
        features = self.net(obs)
        mean = self.mean_NN(features)
        log_std = self.log_std_NN(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class SACAgent(Agent):
    def __init__(
        self,
        env=None,
        name: str = "bot",
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch_size=256,
        buffer_capacity=100_000,
        target_entropy=None,
        alpha_lr=3e-4,
    ):
        super().__init__(env=env, name=name)
        self.actor = SACActor(self.obs_dim, self.action_dim).to(self.device)
        self.critic1 = SACCritic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = SACCritic(self.obs_dim, self.action_dim).to(self.device)
        self.tgt_critic1 = copy.deepcopy(self.critic1)
        self.tgt_critic2 = copy.deepcopy(self.critic2)

        for net in [self.tgt_critic1, self.tgt_critic2]:
            for param in net.parameters():
                param.requires_grad = False

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.target_entropy = target_entropy if target_entropy is not None else -float(self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def act(self, obs, deterministic=False):
        obs_t = self.obs_tensor(obs)
        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(obs_t)
                raw = torch.tanh(mean)
            else:
                raw, _ = self.actor.sample(obs_t)
        self.actor.train()

        action = self.rescale(raw).cpu().numpy()[0]
        return self.clip_action(action)

    def sac_update(self):
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_raw, next_log_prob = self.actor.sample(next_obs)
            next_actions = self.rescale(next_raw).to(self.device).float()
            q1_next = self.tgt_critic1(next_obs, next_actions)
            q2_next = self.tgt_critic2(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1.0 - dones) * (q_next - alpha * next_log_prob)

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

        raw, log_prob = self.actor.sample(obs)
        actor_actions = self.rescale(raw).to(self.device).float()
        actor_actions = torch.max(torch.min(actor_actions, self.action_high), self.action_low)
        q1_actor = self.critic1(obs, actor_actions)
        q2_actor = self.critic2(obs, actor_actions)
        min_q = torch.min(q1_actor, q2_actor)
        actor_loss = (alpha * log_prob - min_q).mean()

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for main, target in [(self.critic1, self.tgt_critic1), (self.critic2, self.tgt_critic2)]:
            for p_main, p_tgt in zip(main.parameters(), target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau)
                p_tgt.data.add_(self.tau * p_main.data)

    def train(self, total_timesteps=200_000, learning_starts=1_000, log_every=20, render=True):
        obs, _ = self.env.reset()
        episode_reward, episode_count = 0.0, 0

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
                        f"Reward: {episode_reward:>8.2f} | Alpha: {self.log_alpha.exp().item():.4f}"
                    )
                episode_reward = 0.0
                obs, _ = self.env.reset()

            if timestep >= learning_starts and len(self.replay_buffer) >= self.batch_size:
                self.sac_update()

        print("Training complete.")
        return self
