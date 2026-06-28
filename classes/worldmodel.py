import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Buffer:
    def __init__(self, env : gym.Env, sampler=lambda env: env.action_space.sample(), capacity=2000): #max out at capacity of 2k
        self.capacity = capacity
        self.episodes = [] #empty array 
        self.position = 0 #index that the next training episode will occupy
        self.env = env
        self.sampler = sampler #random sampler

    def add(self, episode: dict): 
        if len(self.episodes) < self.capacity: #not yet at capacity 
            self.episodes.append(episode) 
        else: #at capacity
            self.episodes[self.position] = episode #replace at that position 
            self.position = (self.position + 1) % self.capacity #reset position

    def add_many(self, episodes: list): #helper for adding a list of training episodes at once
        for ep in episodes:
            self.add(ep)

    def sample_batch(self, batch_size, seq_len, device="cpu"):
        idxs = np.random.choice(len(self.episodes), size=batch_size, replace=True)
        z_in, a_in, z_target, done_target = [], [], [], []
        for i in idxs:
            ep = self.episodes[i]
            T = ep["a"].shape[0]
            z = ep["z"][:T]            # z_0 ... z_{T-1}
            z_next = ep["z"][1:T + 1]  # z_1 ... z_T
            a = self.normalize_action(ep["a"])
            d = ep["done"]
            z_in.append(Buffer.pad(z, seq_len))
            a_in.append(Buffer.pad(a, seq_len))
            z_target.append(Buffer.pad(z_next, seq_len))
            done_target.append(Buffer.pad(d.reshape(-1, 1), seq_len))
        z_in = torch.tensor(np.stack(z_in), device=device)
        a_in = torch.tensor(np.stack(a_in), device=device)
        z_target = torch.tensor(np.stack(z_target), device=device)
        done_target = torch.tensor(np.stack(done_target), device=device)
        return z_in, a_in, z_target, done_target
    
    def collect_rollouts(self, n_episodes=500): #collect episodes using the random sampler built in
        episodes = []
        for ep in range(n_episodes):
            obs, _ = self.env.reset() # reset() returns (obs, info) -- unpack it
            zs = [np.asarray(obs, dtype=np.float32).reshape(-1)]  # seed with z_0
            actions, dones = [], []  # fresh per episode, not shared across episodes
            done = False  # must exist before the while-loop checks it
            while not done:
                rand_action = self.sampler(self.env)
                next_obs, reward, terminated, truncated, _ = self.env.step(rand_action)
                done = terminated or truncated
                actions.append(np.asarray(rand_action, dtype=np.float32).reshape(-1))
                zs.append(np.asarray(next_obs, dtype=np.float32).reshape(-1))
                dones.append(float(done))
            episodes.append({
                "z": np.array(zs, dtype=np.float32),
                "a": np.array(actions, dtype=np.float32),
                "done": np.array(dones, dtype=np.float32),
            })
        return episodes

    def normalize_action(self, a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        low = self.env.action_space.low.astype(np.float32).reshape(-1)
        high = self.env.action_space.high.astype(np.float32).reshape(-1)
        return (a - low) / (high - low) * 2.0 - 1.0

    @staticmethod
    def pad(arr, length):
        out = np.zeros((length,) + arr.shape[1:], dtype=np.float32)
        n = min(length, arr.shape[0])
        out[:n] = arr[:n]
        return out

    def __len__(self):
        return len(self.episodes)
    
class MDNRNN(nn.Module):
    def __init__(self, env: gym.Env, hidden_size=128, n_gaussians=5):
        super().__init__()
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("Only Box observation spaces are supported.")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("Only continuous Box action spaces are supported.")
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians
        input_dim = self.obs_dim + self.action_dim
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True) 
        # MDN head: for each of K gaussians, need (pi, mu, sigma) per obs_dim
        K = n_gaussians
        self.pi_head = nn.Linear(hidden_size, K)
        self.mu_head = nn.Linear(hidden_size, K * self.obs_dim)
        self.sigma_head = nn.Linear(hidden_size, K * self.obs_dim)
        self.done_head = nn.Linear(hidden_size, 1) # done predictor: simple Bernoulli logit (paper uses a >50% cutoff rather than sampling, "more stable" than sampling from Bernoulli)
        #buffer for storing episodes

    def forward(self, z, a, hidden=None):
        x = torch.cat([z, a], dim=-1)
        out, hidden = self.lstm(x, hidden)  # out: (batch, seq_len, hidden_size)
        K, D = self.n_gaussians, self.obs_dim
        pi = torch.softmax(self.pi_head(out), dim=-1)
        mu = self.mu_head(out).view(*out.shape[:-1], K, D)
        sigma = torch.exp(self.sigma_head(out)).clamp(min=1e-4).view(*out.shape[:-1], K, D)
        done_logit = self.done_head(out)
        return pi, mu, sigma, done_logit, hidden
 
    def sample(self, pi, mu, sigma, temperature=1.0):
        batch = pi.shape[0]
        K, D = mu.shape[1], mu.shape[2]
        logits = torch.log(pi.clamp(min=1e-8)) / temperature # temperature-adjusted mixture weights (softmax with temp) and stds
        pi_t = torch.softmax(logits, dim=-1)
        sigma_t = sigma * np.sqrt(temperature)
        # pick a component per batch row
        comp = torch.multinomial(pi_t, num_samples=1).squeeze(-1)  # (batch,)
        idx = comp.view(batch, 1, 1).expand(batch, 1, D)
        chosen_mu = mu.gather(1, idx).squeeze(1)        # (batch, D)
        chosen_sigma = sigma_t.gather(1, idx).squeeze(1)  # (batch, D)
        eps = torch.randn_like(chosen_mu)
        z_next = chosen_mu + eps * chosen_sigma
        return z_next

    @staticmethod
    def mdn_loss(pi, mu, sigma, target):
        target = target.unsqueeze(2)  # (batch, seq_len, 1, D)
        # per-component, per-dim log prob, summed over D (factored gaussian,
        log_prob = -0.5 * (((target - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)  # (batch, seq_len, K)
        log_pi = torch.log(pi.clamp(min=1e-8))
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # (batch, seq_len)
        return -log_mix.mean()
 
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
class DreamEnv(gym.Env):
    def __init__(self, env: gym.Env, calc_reward : callable, max_steps, capacity=2000, hidden_size=128, n_gaussians=5,
                 temperature=1.0, device="cpu", done_threshold=0.5, mdnloss_wt = 1, doneloss_wt = 1):
        self.env = env
        self.mdnrnn = MDNRNN(env, hidden_size=hidden_size, n_gaussians=n_gaussians)
        self.buffer = Buffer(env, capacity=capacity)
        self.device = device
        self.temperature = temperature
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
        self.obs_low = env.observation_space.low.astype(np.float32).reshape(-1)
        self.obs_high = env.observation_space.high.astype(np.float32).reshape(-1)
        self.initial_z = self.infer_initial_z(env)
        self.max_steps = max_steps
        self.done_threshold = done_threshold
        self.hidden = None
        self.z = None
        self.step_count = 0
        self.calc_reward = calc_reward
        self.mdnloss_wt = mdnloss_wt
        self.doneloss_wt = doneloss_wt

    def train(self, n_epochs=20, batch_size=32, seq_len=29, lr=1e-3,
              batches_per_epoch=50, render=False):
        if len(self.buffer) == 0:
            raise RuntimeError("DreamEnv.train() called with an empty buffer. Call collect_rollouts(...) and self.buffer.add_many(...) first.")
        self.mdnrnn.to(self.device)
        self.mdnrnn.train()
        optimizer = optim.Adam(self.mdnrnn.parameters(), lr=lr)
        bce = torch.nn.BCEWithLogitsLoss()
        for epoch in range(n_epochs):
            epoch_mdn_loss, epoch_done_loss = 0.0, 0.0
            for _ in range(batches_per_epoch):
                z_in, a_in, z_target, done_target = self.buffer.sample_batch(
                    batch_size, seq_len, self.device)
                hidden = self.mdnrnn.init_hidden(z_in.shape[0], self.device)
                pi, mu, sigma, done_logit, _ = self.mdnrnn(z_in, a_in, hidden)
                loss_mdn = self.mdnrnn.mdn_loss(pi, mu, sigma, z_target)
                loss_done = bce(done_logit, done_target)
                loss = self.mdnloss_wt*loss_mdn + self.doneloss_wt*loss_done
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mdnrnn.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_mdn_loss += loss_mdn.item()
                epoch_done_loss += loss_done.item()
            if render:
                print(f"Epoch {epoch+1:>3}/{n_epochs} | "
                      f"MDN NLL: {epoch_mdn_loss/batches_per_epoch:.4f} | "
                      f"Done BCE: {epoch_done_loss/batches_per_epoch:.4f}")
        return self

    def reset(self, seed=None, options=None):
        self.mdnrnn.eval()
        self.hidden = self.mdnrnn.init_hidden(1, self.device)
        self.z = self.initial_z.copy()
        self.step_count = 0
        return self.z.reshape(self.obs_shape).copy(), {}
 
    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a_norm = self.buffer.normalize_action(a)
        z_prev = self.z.copy()
        z_t = torch.tensor(self.z, dtype=torch.float32, device=self.device).view(1, 1, -1)
        a_t = torch.tensor(a_norm, dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            pi, mu, sigma, done_logit, self.hidden = self.mdnrnn(z_t, a_t, self.hidden)
            z_next = self.mdnrnn.sample(pi[:, 0], mu[:, 0], sigma[:, 0], temperature=self.temperature)
            done_prob = torch.sigmoid(done_logit[0, 0, 0]).item()
        z_next = z_next.squeeze(0).cpu().numpy()
        z_next = np.clip(z_next, self.obs_low, self.obs_high)
        reward = self.calc_reward(
            z_prev.reshape(self.obs_shape),
            np.asarray(action, dtype=np.float32).reshape(self.action_shape),
            z_next.reshape(self.obs_shape),
            self.env,
        )
        self.step_count += 1
        terminated = done_prob > self.done_threshold
        truncated = self.step_count >= self.max_steps
        self.z = z_next
        return self.z.reshape(self.obs_shape).copy(), reward, terminated, truncated, {}
 
    def obs(self):
        return self.z.reshape(self.obs_shape).copy()
    
    @staticmethod
    def infer_initial_z(env): #helper to create initial state of the environment
        obs, _ = env.reset()
        return np.array(obs, dtype=np.float32).reshape(-1)
    
