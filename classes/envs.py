import gymnasium as gym
import numpy as np


class DynamicPricingEnv(gym.Env):
    def __init__(self):
        self.max_steps = 30
        self.max_inventory = 100
        self.cost = 5.0
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.max_inventory, self.max_steps, self.max_inventory]),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=5.0, high=50.0, shape=(1,), dtype=np.float32)
        self.latest = np.array([self.max_inventory, self.max_steps, 0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = self.max_inventory
        self.step_count = 0
        self.last_demand = 0
        self.latest = self.obs()
        return self.obs(), {}

    def step(self, action):
        price = float(action[0])
        demand = self.demand(price)
        units_sold = min(demand, self.inventory)
        self.inventory -= units_sold
        self.step_count += 1
        self.last_demand = units_sold

        reward = (price - self.cost) * units_sold / 100.0
        terminated = self.inventory <= 0
        truncated = self.step_count >= self.max_steps
        if truncated and self.inventory > 0:
            reward -= self.inventory * 2.0

        self.latest = self.obs()
        return self.obs(), reward, terminated, truncated, {}

    def obs(self):
        return np.array(
            [
                self.inventory / self.max_inventory,
                (self.max_steps - self.step_count) / self.max_steps,
                self.last_demand / self.max_inventory,
            ],
            dtype=np.float32,
        )

    def get_latest(self):
        obs = self.latest
        print(f"Leftover Stock: {obs[0]} units, Days Left {obs[1]}, Sold Units: {obs[2]}")

    def demand(self, price):
        base = 40
        sensitivity = 0.8
        noise = np.random.normal(0, 2)
        return int(max(0, base - sensitivity * price + noise))
