import argparse
import os
import random
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
import tensorflow as tf
import yaml
import gymnasium as gym
import gymnasium_stag_hunt
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.mappo_agent import MAPPOAgent
from agents.random_agent import RandomAgent
from training.trainer import Trainer
from training.mappo_trainer import MAPPOTrainer

AGENT_CLASSES = {
    'a2c': (A2CAgent, 'a2c'),
    'ppo': (PPOAgent, 'ppo'),
    'mappo': (MAPPOAgent, 'mappo'),
    'random': (RandomAgent, None),
}

def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=AGENT_CLASSES.keys(), required=True,
                        help='Agent type to train: a2c | ppo | random')
    parser.add_argument('--game', choices=['Hunt', 'Harvest', 'Escalation'], default='Hunt',
                        help='Environment to train on (default: Hunt)')
    parser.add_argument('--robust', action='store_true',
                        help='Domain randomization: train on both stag_follows=True and False (Hunt only)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Fine-tune standard weights on stag_follows=False only (Hunt only)')
    parser.add_argument('--update', action='store_true',
                        help='Resume training from existing saved weights (if present)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override n_episodes from config')
    args = parser.parse_args()

    with open('configs/default_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if args.episodes is not None:
        config['training']['n_episodes'] = args.episodes

    game = args.game
    config['env']['game'] = game
    game_prefix = game.lower()
    supports_stag = (game == 'Hunt')

    env_kwargs = dict(
        grid_size=tuple(config['env']['grid_size']),
        obs_type=config['env']['obs_type'],
        max_timesteps=config['env']['max_timesteps'],
        enable_multiagent=config['env']['enable_multiagent'],
        flip_obs=config['env']['flip_obs']
    )

    if args.adaptive:
        if supports_stag:
            envs = [gym.make(f"StagHunt-{game}-v0", **env_kwargs, stag_follows=False)]
        else:
            envs = [gym.make(f"StagHunt-{game}-v0", **env_kwargs)]
        model_name = f'{game_prefix}_agent_adaptive'
    elif args.robust:
        if supports_stag:
            envs = [
                gym.make(f"StagHunt-{game}-v0", **env_kwargs, stag_follows=True),
                gym.make(f"StagHunt-{game}-v0", **env_kwargs, stag_follows=False),
            ]
        else:
            print(f"Warning: --robust has no effect for {game} (no stag_follows). Training on single env.")
            envs = [gym.make(f"StagHunt-{game}-v0", **env_kwargs)]
        model_name = f'{game_prefix}_agent_robust'
    else:
        envs = [gym.make(f"StagHunt-{game}-v0", **env_kwargs)]
        model_name = f'{game_prefix}_agent'

    for i, env in enumerate(envs):
        env.reset(seed=42 + i)
    obs, _ = envs[0].reset(seed=42)
    obs_dim = obs[0].shape[0]
    action_dim = envs[0].action_space.n

    AgentClass, config_key = AGENT_CLASSES[args.agent]
    agent_config = config[config_key] if config_key else {}

    if args.agent == 'mappo':
        global_obs_dim = obs_dim  # obs[0] already encodes the full world state
        shared_agent = MAPPOAgent(obs_dim, action_dim, global_obs_dim, agent_config)
        agent_view2 = MAPPOAgent(obs_dim, action_dim, global_obs_dim, agent_config)
    else:
        shared_agent = AgentClass(obs_dim, action_dim, agent_config)
        agent_view2 = AgentClass(obs_dim, action_dim, agent_config)

    agent_view2.actor = shared_agent.actor
    agent_view2.critic = shared_agent.critic
    agent_view2.optimizer = shared_agent.optimizer
    agents = [shared_agent, agent_view2]

    TrainerClass = MAPPOTrainer if args.agent == 'mappo' else Trainer

    if args.adaptive:
        results_dir = config['logging']['results_dir']
        if args.update:
            trainer = TrainerClass(agents, envs, config, model_name=model_name, update=True, agent_tag=args.agent)
        else:
            try:
                shared_agent.load(f"{results_dir}/{game_prefix}_agent_0")
                print(f"Loaded standard {game} weights for adaptive fine-tuning")
            except Exception as e:
                print(f"Warning: could not load standard weights: {e}")
            trainer = TrainerClass(agents, envs, config, model_name=model_name, update=False, agent_tag=args.agent)
    else:
        trainer = TrainerClass(agents, envs, config, model_name=model_name, update=args.update, agent_tag=args.agent)

    trainer.train()
    print(f"Rewards saved to {trainer.rewards_path}")
    for env in envs:
        env.close()

if __name__ == "__main__":
    main()

