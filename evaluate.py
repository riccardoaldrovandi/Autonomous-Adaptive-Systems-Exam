import argparse
import yaml
import gymnasium as gym
import gymnasium_stag_hunt
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.mappo_agent import MAPPOAgent
from agents.random_agent import RandomAgent
from training.evaluator import Evaluator

AGENT_CLASSES = {
    'a2c': (A2CAgent, 'a2c'),
    'ppo': (PPOAgent, 'ppo'),
    'mappo': (MAPPOAgent, 'mappo'),
    'random': (RandomAgent, None),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=AGENT_CLASSES.keys(), required=True,
                        help='Agent type to evaluate: a2c | ppo | random')
    parser.add_argument('--game', choices=['Hunt', 'Harvest', 'Escalation'], default='Hunt',
                        help='Environment to evaluate on (default: Hunt)')
    parser.add_argument('--robust', action='store_true',
                        help='Evaluate robust model (trained with domain randomization)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Evaluate adaptive model (fine-tuned on stag_follows=False)')
    parser.add_argument('--evo', action='store_true',
                        help='Evaluate evolutionary-optimized model (trained with train_evo.py)')
    parser.add_argument('--stag-follows', action='store_true', default=True,
                        help='Whether stag follows agents, Hunt only (default True)')
    parser.add_argument('--no-stag-follows', dest='stag_follows', action='store_false',
                        help='Stag moves randomly (Hunt only)')
    args = parser.parse_args()

    with open('configs/default_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    game = args.game
    config['env']['game'] = game
    game_prefix = game.lower()
    supports_stag = (game == 'Hunt')

    env_kwargs = dict(
        grid_size=tuple(config['env']['grid_size']),
        obs_type=config['env']['obs_type'],
        max_timesteps=config['env']['max_timesteps'],
        enable_multiagent=config['env']['enable_multiagent'],
        flip_obs=config['env']['flip_obs'],
    )
    if supports_stag:
        env_kwargs['stag_follows'] = False if args.adaptive else args.stag_follows

    env = gym.make(f"StagHunt-{game}-v0", **env_kwargs)

    obs, _ = env.reset()
    obs_dim = obs[0].shape[0]
    action_dim = env.action_space.n

    AgentClass, config_key = AGENT_CLASSES[args.agent]
    agent_config = config[config_key] if config_key else {}
    results_dir = config['logging']['results_dir']

    if args.adaptive:
        model_name = f'{game_prefix}_agent_adaptive'
    elif args.evo and args.robust:
        model_name = f'{game_prefix}_{args.agent}_evo_robust'
    elif args.evo:
        model_name = f'{game_prefix}_{args.agent}_evo'
    elif args.robust:
        model_name = f'{game_prefix}_agent_robust'
    else:
        model_name = f'{game_prefix}_agent'

    if args.agent == 'mappo':
        shared_agent = MAPPOAgent(obs_dim, action_dim, obs_dim, agent_config)
    else:
        shared_agent = AgentClass(obs_dim, action_dim, agent_config)
    shared_agent.load(f"{results_dir}/{model_name}_0")
    agents = [shared_agent, shared_agent]

    evaluator = Evaluator(agents, env, config, game=game)
    evaluator.evaluate()
    env.close()


if __name__ == "__main__":
    main()
