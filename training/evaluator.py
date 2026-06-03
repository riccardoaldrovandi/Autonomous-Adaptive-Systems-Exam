import numpy as np
from agents.base_agent import BaseAgent


class Evaluator:
    def __init__(self, agents: list[BaseAgent], env, config: dict, game: str = 'Hunt'):
        self.agents = agents
        self.env = env
        self.n_episodes = config['eval']['n_episodes']
        self.render = config['eval']['render']
        self.game = game

    def _is_cooperative_event(self, rewards) -> bool:
        if self.game == 'Hunt':
            return rewards[0] >= 5.0 and rewards[1] >= 5.0
        elif self.game == 'Harvest':
            return rewards[0] >= 2.0 and rewards[1] >= 2.0
        elif self.game == 'Escalation':
            return rewards[0] >= 1.0 and rewards[1] >= 1.0
        return False

    def _is_solo_event(self, rewards) -> bool:
        if self.game == 'Hunt':
            return rewards[0] == 1.0 or rewards[1] == 1.0
        return False

    def evaluate(self) -> dict:
        episode_rewards = []
        episode_coop_events = []
        episode_solo_events = []
        cooperation_episodes = 0

        for episode in range(1, self.n_episodes + 1):
            obs, _ = self.env.reset()
            done = False
            ep_reward = [0.0] * len(self.agents)
            coop_events = 0
            solo_events = 0

            while not done:
                actions = []
                for i, agent in enumerate(self.agents):
                    action, _, _ = agent.act(obs[i], deterministic=True)
                    actions.append(action)

                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated

                if self._is_cooperative_event(rewards):
                    coop_events += 1
                elif self._is_solo_event(rewards):
                    solo_events += 1

                for i in range(len(self.agents)):
                    ep_reward[i] += rewards[i]

                obs = next_obs

            episode_rewards.append(np.sum(ep_reward))
            episode_coop_events.append(coop_events)
            episode_solo_events.append(solo_events)
            if coop_events > 0:
                cooperation_episodes += 1

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_coop = np.mean(episode_coop_events)
        avg_solo = np.mean(episode_solo_events)
        cooperation_rate = cooperation_episodes / self.n_episodes

        coop_label = {'Hunt': 'stag catches', 'Harvest': 'coop harvests', 'Escalation': 'mutual coop'}.get(self.game, 'coop events')
        solo_label = {'Hunt': 'plant catches'}.get(self.game, 'solo events')

        print(f"Evaluation over {self.n_episodes} episodes ({self.game}):")
        print(f"  Avg total reward    : {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Min / Max           : {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
        print(f"  Avg {coop_label:<16}: {avg_coop:.2f} per episode")
        if self.game == 'Hunt':
            print(f"  Avg {solo_label:<16}: {avg_solo:.2f} per episode")
        print(f"  Cooperation rate    : {cooperation_rate * 100:.1f}% of episodes")

        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'episode_rewards': episode_rewards,
            'avg_coop_events': avg_coop,
            'avg_solo_events': avg_solo,
            'cooperation_rate': cooperation_rate
        }
