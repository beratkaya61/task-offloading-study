# Progress Article 04: Reinforcement Learning & PPO Agent Architecture

## 1. Why Reinforcement Learning (RL)?
Rule-based systems (like our current Greedy or Semantic logic) are "stateless" and "static". They don't learn from past failures. In a complex IoT environment:
- Network conditions fluctuate wildly.
- Server queues vary stochastically.
- Deadlines and energy constraints compete.

**Deep Reinforcement Learning (DRL)** allows an agent to learn an optimal **Policy ($\pi$)** by interacting with the environment and receiving rewards or penalties. Over thousands of episodes, it learns to predict which action (Local, Edge, Cloud) yields the highest long-term benefit.

## 2. Proximal Policy Optimization (PPO)
We use the **PPO** algorithm because it is:
- **Stable**: Avoids large, destructive policy updates.
- **Sample Efficient**: Learns faster than older methods like DQN or Vanilla Policy Gradient.
- **Robust**: Works well with continuous and discrete action spaces.

## 3. Reward Shaping (Semantic Guidance)
A standard RL agent might take a long time to understand that "Critical tasks must never miss deadlines". 
**Reward Shaping** is the process of fine-tuning the reward function to include domain knowledge. 

We use our **LLM Semantic Analyzer** to "shape" the reward:
$$ Reward = \alpha \cdot (\text{Energy Saved}) + \beta \cdot (\text{Time Saved}) - \gamma \cdot (\text{Penalty}) $$

Where $\gamma$ (Penalty) is **dynamically adjusted by the LLM**:
- If Task = `CRITICAL`, penalty for missing deadline is $100x$.
- If Task = `BEST_EFFORT`, penalty for missing deadline is $1x$.

This "Semantic Reward" guides the PPO agent to prioritize tasks exactly like a human expert would, but with the speed of a machine.

## 4. Implementation Goals
1. Define a custom `gym.Env` for the IoT simulation.
2. Integrate `Stable-Baselines3` for the PPO agent.
3. Train the model using the Google Cluster and Didi Gaia data.

---
*Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.*
