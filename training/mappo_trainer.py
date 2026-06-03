import gc
import random
import numpy as np
from training.trainer import Trainer


class MAPPOTrainer(Trainer):
    """
    Extends Trainer for MAPPO: computes the global state (concatenation of all
    agents' observations) and passes it to act() and store_experience() so the
    centralized critic can use it during training.
    """

    def train(self) -> list[float]:
        for episode in range(1, self.n_episodes + 1):
            env = random.choice(self.envs)
            obs, _ = env.reset()
            done = False
            step_count = 0
            episode_rewards = [0.0] * len(self.agents)

            while not done:
                # obs[0] already contains positions of all agents + stag + plants
                # in canonical form — no need to concatenate with the flipped obs[1]
                global_obs = obs[0]

                actions, log, values = [], [], []
                for i, agent in enumerate(self.agents):
                    action, log_prob, value = agent.act(obs[i], global_state=global_obs)
                    actions.append(action)
                    log.append(log_prob)
                    values.append(value)

                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated
                next_global_obs = next_obs[0]

                for i, agent in enumerate(self.agents):
                    agent.store_experience(
                        obs[i], actions[i], rewards[i], next_obs[i], done,
                        log[i], values[i], global_obs, next_global_obs
                    )
                    episode_rewards[i] += rewards[i]

                obs = next_obs
                step_count += 1

                for i, agent in enumerate(self.agents):
                    if step_count % self.n_steps[i] == 0:
                        agent.update()

            for i, agent in enumerate(self.agents):
                if len(agent.obs_buffer) > 0:
                    agent.update(force=True)

            total_episode_reward = np.sum(episode_rewards)
            self.episode_rewards.append(total_episode_reward)
            gc.collect()

            if episode % self.eval_every == 0:
                avg = np.mean(self.episode_rewards[-self.eval_every:])
                print(f"Episode {episode}/{self.n_episodes}, avg reward (last {self.eval_every} episodes): {avg:.2f}")
                np.save(self.rewards_path, np.array(self.episode_rewards))

                if avg > self.best_reward:
                    self.best_reward = avg
                    if self.save_model:
                        for i, agent in enumerate(self.agents):
                            agent.save(f"{self.results_dir}/{self.model_name}_{i}")

        return self.episode_rewards
