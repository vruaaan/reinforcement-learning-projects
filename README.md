# Reinforcement Learning Projects

This repository documents my journey through Reinforcement Learning during Summer 2026, starting with the default environments inside Python's Gymnasium environment such as blackjack and frozen lake and advancing towards custom environments such as the card game dragontiger and dynamic pricing scenarios.

### **Notebooks:**
  - [frozenlake.ipynb](frozenlake.ipynb) : Q-learning on OpenAI Gym FrozenLake (tabular Q-table and agent class, training + evaluation).
  - [blackjack.ipynb](blackjack.ipynb) : Q-learning agent for the Blackjack environment with policy visualization and learning curve.
  - [dragontiger.ipynb](dragontiger.ipynb) : Custom Dragon-Tiger card game environment and baseline agents (env, deck, rules, rewards).
  - [dynpricing_pytorch.ipynb](dynpricing_pytorch.ipynb) : Dynamic pricing environment + PyTorch implementations (PPO, TD3, SAC components). Includes training/evaluation loops for the PPO, TD3 and SAC models.
  - [dynpricing_pvp.ipynb](dynpricing_pvp.ipynb) : Multi-agent extension that pits PPO/TD3/SAC agents against each other in a shared market environment, simulating an oligopolistic market environment.

___
### **Dependencies (suggested)**
- Python 3.9+
- gymnasium
- numpy
- matplotlib
- torch (if you want to run the PyTorch notebooks)

### **Quickstart**
- Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
```

- Install dependencies using `requirements.txt`:

```powershell
pip install --upgrade pip
pip install requirements.txt
```
- Open the project in Jupyter / VS Code and run the notebooks interactively.

___
### **Notebook contents**
1. **Blackjack** ([blackjack.ipynb](blackjack.ipynb))
  - Tabular Q-learning for Blackjack-v1.
  - Produces a learned policy visualization and learning curve plots.
    
2. **FrozenLake** ([frozenlake.ipynb](frozenlake.ipynb))
  - Demonstrates a tabular Q-learning agent for Gym's FrozenLake-v1 (slippery).
  - Includes a `FrozenLakeAgent` class with `train()` and `evaluate()` methods.
  - Run cells to train (EPISODES configurable) and call `evaluate()` to compute success rate.

3. **DragonTiger** ([dragontiger.ipynb](dragontiger.ipynb))
  - Implements a custom Gym environment with card/deck classes and a configurable game loop, useful for testing bandit-style and RL betting agents.
  - Demonstrates a tabular Q-learning agent for the custom dragontiger environment, utilising double Q-tables
  - Includes a `DragonTigerAgent` class with `train()` and `evaluate()` methods.
  - Simulates real scenarios where the model walks away if it suffers a huge loss or if it has made a decent profit (implemented with the method `should_walk_away`) and keeps track of winstreaks to make decisions as well 
  - Run cells to train (EPISODES configurable) and call `evaluate()` to compute success rate.

4. **Dynamic Pricing (PyTorch)** ([dynpricing_pytorch.ipynb](dynpricing_pytorch.ipynb))
  - Builds a custom continuous-action dynamic-pricing environment that simulates a store owner selling 100 units of goods over a span of 30 days.
  - Since the model is the only seller (model) in the environment, this actually simulates a Monopolistic market.
  - Implements 3 types of models with the respective components
      1. Proximal Policy Optimisation (PPO) -> `class PPOAgent` 
         - PPOActorCritic
         - RolloutBuffer
      3. Twin Delayed Deep Deterministic Policy Gradient(TD3) -> `class TD3Agent`
         - TD3Actor
         - TD3Critic
         - ReplayBuffer
      5. Soft-Actor-Critic(SAC) models -> `class SACAgent`
         - SACActor
         - SACCritic
         - ReplayBuffer (reused from TD3)
  - Each model's class includes a `train()` and `evaluate()` to train the model and evaluate the performance of each model
  - Each model also has a `play()` method that simulates the model's actions in the environment for 1 episode 

5. **Dynamic Pricing PvP** ([dynpricing_pvp.ipynb](dynpricing_pvp.ipynb))
  - Builds a tailored version of the custom dynamic pricing environment `OligopolyEnv`, where the demand signal is a result of all 3 model's actions, ensuring that all models in the environment sees the same state at each step 
  - Includes a seperate `train(agents, n_rounds, timesteps_per_round)` function to run PvP rounds and save round checkpoints as `.pth` files
  - Includes a `play_oligopoly(agents, n_episodes=1, deterministic=True)` function to load the final model's `.pth` file and use them to simulate the model's actions in the environment for 1 or more episode

**Recommended workflow**
- Inspect the notebook for the environment and hyperparameters first (the `1. Imports` / `2. Environment` sections).
- Reduce training budgets (timesteps / episodes) when experimenting locally to keep runtimes reasonable. Use evaluation cells after training to check performance.
