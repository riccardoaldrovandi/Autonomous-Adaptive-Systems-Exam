import gc
import os
import random
import numpy as np
from agents.base_agent import BaseAgent

class Trainer():
    def __init__ (self, agents: list[BaseAgent], env, config: dict, model_name: str = 'agent', update: bool = False, agent_tag: str = ''):

        self.agents = agents
        self.envs = env if isinstance(env, list) else [env]

        self.n_episodes = config['training']['n_episodes']
        self.eval_every = config['training']['eval_every']
        self.results_dir = config['logging']['results_dir']
        self.save_model = config['logging']['save_model']
        self.model_name = model_name
        self.best_reward = -float('inf')

        self.n_steps = [agent.n_steps for agent in self.agents ]

        self.episode_rewards = []

        tag = f'_{agent_tag}' if agent_tag else ''
        self.rewards_path = f"{self.results_dir}/{model_name}{tag}_rewards.npy"

        if update:
            self._load_weights()

    def _load_weights(self):
        for i, agent in enumerate(self.agents):
            path = f"{self.results_dir}/{self.model_name}_{i}"
            try:
                agent.load(path)
                print(f"Loaded weights for agent {i} from '{path}'")
            except Exception:
                print(f"No saved weights found for agent {i} at '{path}', starting from scratch")

    def train(self) -> list[float]:
        for episode in range(1, self.n_episodes + 1):
            env = random.choice(self.envs)
            obs,_ = env.reset()
            done = False
            step_count = 0
            episode_rewards = [0.0] * len(self.agents)
            
            while not done:

                actions,log,values = [], [] , []

                for i, agent in enumerate(self.agents):
                    action, log_prob, value = agent.act(obs[i])
                    actions.append(action)
                    log.append(log_prob)
                    values.append(value)

                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated

                for i, agent in enumerate(self.agents):
                    agent.store_experience(obs[i], actions[i], rewards[i], next_obs[i], done, log[i], values[i])

                    episode_rewards[i] += rewards[i]
                
                obs = next_obs
                step_count += 1

                for i, agent in enumerate(self.agents):
                    if step_count % self.n_steps[i] == 0:
                        agent.update()

            for i, agent in enumerate(self.agents):
                if len(agent.obs_buffer) > 0:
                    agent.update()

            total_episode_reward = np.sum(episode_rewards)
            self.episode_rewards.append(total_episode_reward)

            if episode % self.eval_every == 0:
               avg = np.mean(self.episode_rewards[-self.eval_every:])

               print(f"Episode {episode}/{self.n_episodes}, avg reward (last {self.eval_every} episodes): {avg:.2f} ")
               np.save(self.rewards_path, np.array(self.episode_rewards))
               gc.collect()

               if avg > self.best_reward:
                  self.best_reward = avg
                  if self.save_model:
                    for i, agent in enumerate(self.agents):
                        agent.save(f"{self.results_dir}/{self.model_name}_{i}")

        return self.episode_rewards








