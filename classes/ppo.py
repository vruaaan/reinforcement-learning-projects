import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

try:
    from .agents import Agent
    from .buffers import RolloutBuffer
except ImportError:
    from agents import Agent
    from buffers import RolloutBuffer


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 2.0)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        features = self.backbone(obs)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        critic_val = self.critic(features)
        return mean, std, critic_val

    def get_action(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy


class PPOAgent(Agent):
    def __init__(self, env=None, name: str = "bot", network=None, lr=3e-4):
        super().__init__(env=env, name=name)
        self.NN = (
            PPOActorCritic(self.obs_dim, self.action_dim).to(self.device)
            if network is None
            else network.to(self.device)
        )
        self.buffer = RolloutBuffer()
        self.optimizer = optim.Adam(self.NN.parameters(), lr=lr)

    def train(self, total_timesteps=200_000, n_steps=512, render=True):
        self.NN.train()
        obs, _ = self.env.reset()
        episode_reward, episode_count = 0.0, 0
        timestep = 0

        while timestep < total_timesteps:
            self.buffer.clear()
            for _ in range(n_steps):
                obs_tensor = self.obs_tensor(obs)
                with torch.no_grad():
                    action, log_prob, value = self.NN.get_action(obs_tensor)

                action_np = self.clip_action(action.cpu().numpy()[0])
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.add(
                    obs=obs,
                    action=action.squeeze(0).cpu(),
                    log_prob=log_prob.squeeze(0).cpu(),
                    reward=reward,
                    value=value.squeeze(0).cpu().item(),
                    done=float(done),
                )

                episode_reward += reward
                obs = next_obs
                timestep += 1

                if done:
                    episode_count += 1
                    if episode_count % 20 == 0 and render:
                        print(
                            f"Timestep {timestep:>7} | Episode {episode_count:>4} | "
                            f"Reward: {episode_reward:>8.2f}"
                        )
                    episode_reward = 0.0
                    obs, _ = self.env.reset()

                if timestep >= total_timesteps:
                    break

            with torch.no_grad():
                last_obs = self.obs_tensor(obs)
                _, _, last_value = self.NN.get_action(last_obs)
                last_value = last_value.squeeze(0).cpu().item()

            advantages, returns = self.buffer.compute_returns(last_value)
            obs_t, act_t, lp_t, adv_t, ret_t = self.buffer.to_tensors(
                advantages, returns, self.device
            )
            self.ppo_update(obs_t, act_t, lp_t, adv_t, ret_t)

        print("Training complete.")
        return self

    def act(self, obs, deterministic=True):
        obs_tensor = self.obs_tensor(obs)
        self.NN.eval()
        with torch.no_grad():
            mean, std, _ = self.NN.forward(obs_tensor)
            action = mean if deterministic else Normal(mean, std).sample()
        action_np = action.squeeze(0).cpu().numpy()
        return self.clip_action(action_np)

    def ppo_update(self, obs, actions, old_log_probs, advantages, returns, clip_range=0.2, ent_coef=0.05, vf_coef=0.5, n_epochs=10, batch_size=64):
        total_steps = obs.shape[0]
        self.NN.train()
        for _ in range(n_epochs):
            indices = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, batch_size):
                idx = indices[start : start + batch_size]
                new_log_probs, values, entropy = self.NN.evaluate(obs[idx], actions[idx])
                ratio = (new_log_probs - old_log_probs[idx]).exp()
                adv = advantages[idx]
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv,
                ).mean()
                value_loss = nn.functional.mse_loss(values, returns[idx])
                entropy_loss = -entropy.mean()
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.NN.parameters(), max_norm=0.5)
                self.optimizer.step()
