import argparse
import copy
import os
import random
import numpy as np
import yaml
import gymnasium as gym
import gymnasium_stag_hunt
import tensorflow.keras.backend as K

from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.mappo_agent import MAPPOAgent
from training.trainer import Trainer
from training.mappo_trainer import MAPPOTrainer

AGENT_CLASSES = {
    'a2c':   (A2CAgent,   'a2c'),
    'ppo':   (PPOAgent,   'ppo'),
    'mappo': (MAPPOAgent, 'mappo'),
}

# (scale, min, max)
# 'log'    → sampled/perturbed in log space (lr, entropy_coef)
# 'linear' → sampled/perturbed in linear space
HYPERPARAM_SPACE = {
    'a2c': {
        'lr':           ('log',    1e-4, 1e-2),
        'gamma':        ('linear', 0.90, 0.999),
        'entropy_coef': ('log',    0.01, 0.30),
        'value_coef':   ('linear', 0.30, 0.70),
    },
    'ppo': {
        'lr':           ('log',    1e-4, 1e-3),
        'gamma':        ('linear', 0.90, 0.999),
        'entropy_coef': ('log',    1e-3, 0.05),
        'value_coef':   ('linear', 0.30, 0.70),
        'lam':          ('linear', 0.90, 0.99),
        'eps_clip':     ('linear', 0.10, 0.30),
    },
    'mappo': {
        'lr':           ('log',    1e-4, 1e-3),
        'gamma':        ('linear', 0.90, 0.999),
        'entropy_coef': ('log',    1e-3, 0.05),
        'value_coef':   ('linear', 0.30, 0.70),
        'lam':          ('linear', 0.90, 0.99),
        'eps_clip':     ('linear', 0.10, 0.30),
    },
}

MUTATION_FACTORS = [0.8, 1.2]


# ── Hyperparameter operators ──────────────────────────────────────────────────

def sample_hyperparams(agent_key: str) -> dict:
    hparams = {}
    for key, (scale, lo, hi) in HYPERPARAM_SPACE[agent_key].items():
        if scale == 'log':
            val = np.exp(np.random.uniform(np.log(lo), np.log(hi)))
        else:
            val = np.random.uniform(lo, hi)
        hparams[key] = float(np.clip(val, lo, hi))
    return hparams


def crossover(parent1: dict, parent2: dict, agent_key: str) -> dict:
    """Uniform crossover: each gene independently drawn from one of the two parents."""
    child = {}
    for key in HYPERPARAM_SPACE[agent_key]:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


def mutate(hparams: dict, agent_key: str, mutation_prob: float = 0.5) -> dict:
    """Each gene mutated independently with probability mutation_prob."""
    space = HYPERPARAM_SPACE[agent_key]
    mutated = {}
    for key, val in hparams.items():
        _, lo, hi = space[key]
        if random.random() < mutation_prob:
            val = float(np.clip(val * random.choice(MUTATION_FACTORS), lo, hi))
        mutated[key] = val
    return mutated


# ── Agent helpers ─────────────────────────────────────────────────────────────

def build_agents(AgentClass: type, agent_key: str,
                 obs_dim: int, action_dim: int,
                 hparams: dict, base_config: dict) -> list:
    """Build a fresh agent pair with random weight initialization."""
    agent_config = copy.deepcopy(base_config.get(agent_key, {}))
    agent_config.update(hparams)

    if agent_key == 'mappo':
        shared = AgentClass(obs_dim, action_dim, obs_dim, agent_config)
        view2  = AgentClass(obs_dim, action_dim, obs_dim, agent_config)
    else:
        shared = AgentClass(obs_dim, action_dim, agent_config)
        view2  = AgentClass(obs_dim, action_dim, agent_config)

    view2.actor     = shared.actor
    view2.critic    = shared.critic
    view2.optimizer = shared.optimizer
    return [shared, view2]


def train_one_generation(agents: list, envs: list, base_config: dict,
                         agent_key: str, n_episodes: int) -> float:
    """Train agents for n_episodes and return fitness (avg reward over last 20%)."""
    gen_config = copy.deepcopy(base_config)
    gen_config['training']['n_episodes'] = n_episodes
    gen_config['logging']['save_model'] = False
    gen_config['training']['eval_every'] = n_episodes + 1

    TrainerClass = MAPPOTrainer if agent_key == 'mappo' else Trainer
    trainer = TrainerClass(agents, envs, gen_config, model_name='_evo_tmp', update=False)
    rewards = trainer.train()

    tail = n_episodes if n_episodes <= 50 else max(1, n_episodes // 5)
    return float(np.mean(rewards[-tail:]))


# ── Evolutionary controller ───────────────────────────────────────────────────

class EvoController:
    """
    Generational evolutionary algorithm for hyperparameter search.

    Each generation:
      1. All individuals trained for episodes_per_gen episodes
      2. Top elite_k individuals survive unchanged (elitism)
      3. Remaining slots filled via crossover + mutation from top parent_pool_size parents
         Children always start from random weight initialization (pure hparam signal)
    """

    def __init__(self, agent_key: str, obs_dim: int, action_dim: int,
                 envs: list, base_config: dict,
                 pop_size: int, n_generations: int, episodes_per_gen: int,
                 elite_k: int, parent_pool_size: int):
        self.agent_key        = agent_key
        self.obs_dim          = obs_dim
        self.action_dim       = action_dim
        self.envs             = envs
        self.base_config      = base_config
        self.pop_size         = pop_size
        self.n_generations    = n_generations
        self.episodes_per_gen = episodes_per_gen
        self.elite_k          = elite_k
        self.parent_pool_size = parent_pool_size

        AgentClass, _ = AGENT_CLASSES[agent_key]
        self.AgentClass = AgentClass

        print(f"Initializing population ({pop_size} individuals)...")
        self.population = []
        for _ in range(pop_size):
            hparams = sample_hyperparams(agent_key)
            agents  = build_agents(AgentClass, agent_key, obs_dim, action_dim,
                                   hparams, base_config)
            self.population.append({'agents': agents, 'hparams': hparams,
                                    'fitness': -float('inf')})

    def _fresh_agents(self, hparams: dict) -> list:
        return build_agents(self.AgentClass, self.agent_key,
                            self.obs_dim, self.action_dim,
                            hparams, self.base_config)

    def _save_checkpoint(self, gen: int, results_dir: str, model_name: str,
                         best_fitness: float, best_hparams: dict) -> None:
        ckpt_data = {
            'completed_gen': gen,
            'best_fitness': float(best_fitness),
            'best_hparams': best_hparams,
            'population': [
                {'hparams': m['hparams'], 'fitness': float(m['fitness'])}
                for m in self.population
            ],
        }
        for idx, member in enumerate(self.population):
            member['agents'][0].save(f"{results_dir}/{model_name}_ckpt_{idx}")
        with open(f"{results_dir}/{model_name}_checkpoint.yaml", 'w') as f:
            yaml.dump(ckpt_data, f, default_flow_style=False)
        print(f"  [checkpoint saved: generation {gen} complete]")

    def _load_from_checkpoint(self, results_dir: str, model_name: str):
        ckpt_path = f"{results_dir}/{model_name}_checkpoint.yaml"
        if not os.path.exists(ckpt_path):
            return None
        with open(ckpt_path, 'r') as f:
            ckpt = yaml.safe_load(f)
        print(f"Checkpoint found — resuming from generation {ckpt['completed_gen'] + 1}.")
        population = []
        for idx, entry in enumerate(ckpt['population']):
            agents = self._fresh_agents(entry['hparams'])
            agents[0].load(f"{results_dir}/{model_name}_ckpt_{idx}")
            population.append({'agents': agents,
                                'hparams': entry['hparams'],
                                'fitness': entry['fitness']})
        return {
            'population':   population,
            'best_fitness': ckpt['best_fitness'],
            'best_hparams': ckpt['best_hparams'],
            'start_gen':    ckpt['completed_gen'] + 1,
        }

    def run(self, results_dir: str, model_name: str):
        best_fitness = -float('inf')
        best_hparams = None
        start_gen = 1

        ckpt = self._load_from_checkpoint(results_dir, model_name)
        if ckpt is not None:
            self.population = ckpt['population']
            best_fitness    = ckpt['best_fitness']
            best_hparams    = ckpt['best_hparams']
            start_gen       = ckpt['start_gen']

        for gen in range(start_gen, self.n_generations + 1):
            print(f"\n{'='*60}")
            print(f"Generation {gen}/{self.n_generations}  "
                  f"({self.episodes_per_gen} episodes per individual)")
            print(f"{'='*60}")

            # ── Step 1: evaluate all ──────────────────────────────────────────
            for idx, member in enumerate(self.population):
                fitness = train_one_generation(
                    member['agents'], self.envs, self.base_config,
                    self.agent_key, self.episodes_per_gen,
                )
                member['fitness'] = fitness
                hp_str = '  '.join(
                    f"{k}={v:.2e}" if v < 0.01 else f"{k}={v:.4f}"
                    for k, v in member['hparams'].items()
                )
                print(f"  [{idx:2d}] fitness={fitness:8.3f}  {hp_str}")

            ranked = sorted(self.population, key=lambda m: m['fitness'], reverse=True)

            # ── Step 2: track global best ─────────────────────────────────────
            if ranked[0]['fitness'] > best_fitness:
                best_fitness = ranked[0]['fitness']
                best_hparams = copy.deepcopy(ranked[0]['hparams'])
                for i, agent in enumerate(ranked[0]['agents']):
                    agent.save(f"{results_dir}/{model_name}_{i}")
                hparams_path = f"{results_dir}/{model_name}_best_hparams.yaml"
                with open(hparams_path, 'w') as f:
                    yaml.dump(best_hparams, f, default_flow_style=False)
                print(f"\n  *** New best: {best_fitness:.3f} — weights and hparams saved ***")

            if gen == self.n_generations:
                break

            # ── Step 3: build next generation ────────────────────────────────
            next_pop = []

            # Elitism: top elite_k survive with their current weights
            for elite in ranked[:self.elite_k]:
                next_pop.append(elite)

            # Parent pool restricted to top parent_pool_size individuals
            parent_pool = ranked[:self.parent_pool_size]

            # Fill remaining slots: crossover + mutation, fresh random weights
            while len(next_pop) < self.pop_size:
                p1 = random.choice(parent_pool)
                p2 = random.choice(parent_pool)
                while p2 is p1:
                    p2 = random.choice(parent_pool)

                child_hparams = crossover(p1['hparams'], p2['hparams'], self.agent_key)
                child_hparams = mutate(child_hparams, self.agent_key)

                next_pop.append({'agents': self._fresh_agents(child_hparams),
                                 'hparams': child_hparams,
                                 'fitness': -float('inf')})

            self.population = next_pop
            self._save_checkpoint(gen, results_dir, model_name, best_fitness, best_hparams)

        print(f"\nEvolution complete.")
        print(f"Best fitness : {best_fitness:.3f}")
        print(f"Best hparams : {best_hparams}")
        return best_hparams, best_fitness


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evolutionary hyperparameter search for RL agents')
    parser.add_argument('--agent', choices=AGENT_CLASSES.keys(), default='a2c',
                        help='Agent type (default: a2c)')
    parser.add_argument('--game', choices=['Hunt', 'Harvest', 'Escalation'], default='Hunt',
                        help='Environment (default: Hunt)')
    parser.add_argument('--pop-size', type=int, default=8,
                        help='Population size (default: 8)')
    parser.add_argument('--n-generations', type=int, default=50,
                        help='Number of generations (default: 50)')
    parser.add_argument('--episodes-per-gen', type=int, default=100,
                        help='Training episodes per individual per generation (default: 100)')
    parser.add_argument('--elite-k', type=int, default=2,
                        help='Elite individuals preserved each generation (default: 2)')
    parser.add_argument('--parent-pool', type=int, default=3,
                        help='Top individuals used as parents for crossover (default: 3)')
    parser.add_argument('--robust', action='store_true',
                        help='Domain randomization: train on both stag conditions (Hunt only)')
    parser.add_argument('--no-stag-follows', action='store_true',
                        help='Use stag_follows=False (Hunt only)')
    args = parser.parse_args()

    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    game_prefix   = args.game.lower()
    supports_stag = (args.game == 'Hunt')

    env_kwargs = dict(
        grid_size=tuple(config['env']['grid_size']),
        obs_type=config['env']['obs_type'],
        max_timesteps=config['env']['max_timesteps'],
        enable_multiagent=config['env']['enable_multiagent'],
        flip_obs=config['env']['flip_obs'],
    )

    if args.robust and supports_stag:
        envs = [
            gym.make(f"StagHunt-{args.game}-v0", **env_kwargs, stag_follows=True),
            gym.make(f"StagHunt-{args.game}-v0", **env_kwargs, stag_follows=False),
        ]
        model_name = f"{game_prefix}_{args.agent}_evo_robust"
    elif args.no_stag_follows and supports_stag:
        envs = [gym.make(f"StagHunt-{args.game}-v0", **env_kwargs, stag_follows=False)]
        model_name = f"{game_prefix}_{args.agent}_evo"
    else:
        envs = [gym.make(f"StagHunt-{args.game}-v0", **env_kwargs)]
        model_name = f"{game_prefix}_{args.agent}_evo"

    obs, _     = envs[0].reset()
    obs_dim    = obs[0].shape[0]
    action_dim = envs[0].action_space.n

    total_ep = args.pop_size * args.n_generations * args.episodes_per_gen
    print(f"Agent : {args.agent}   Game : {args.game}   Model : {model_name}")
    print(f"Budget: {args.pop_size} individuals × {args.n_generations} generations "
          f"× {args.episodes_per_gen} ep = ~{total_ep:,} episodes total")
    print(f"Elitism: {args.elite_k}   Parent pool: top {args.parent_pool}")

    controller = EvoController(
        agent_key=args.agent,
        obs_dim=obs_dim,
        action_dim=action_dim,
        envs=envs,
        base_config=config,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        episodes_per_gen=args.episodes_per_gen,
        elite_k=args.elite_k,
        parent_pool_size=args.parent_pool,
    )

    controller.run(config['logging']['results_dir'], model_name)

    for env in envs:
        env.close()


if __name__ == '__main__':
    main()
