import argparse
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
import gymnasium_stag_hunt

# Agent imports are deferred inside main() so that pygame.init() (triggered by
# load_renderer=True) runs before TensorFlow is imported, avoiding the
# TF+SDL segfault on Linux.

AGENT_KEYS = ['a2c', 'ppo', 'mappo', 'random']
AGENT_CONFIG_KEY = {'a2c': 'a2c', 'ppo': 'ppo', 'mappo': 'mappo', 'random': None}

ENVIRONMENTS = ['Hunt', 'Harvest', 'Escalation']


def draw_state(ax, obs, grid_size, episode, step, total_reward, env_name):
    ax.clear()
    W, H = grid_size

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.grid(True, color='#cccccc', linewidth=0.5)
    ax.set_facecolor('#f9f9f9')
    ax.set_title(f'[{env_name}]  Episode {episode}  —  Step {step}  —  Reward {total_reward:.0f}',
                 fontsize=11)

    coords = obs[0]
    xa, ya = int(coords[0]), int(coords[1])
    xb, yb = int(coords[2]), int(coords[3])

    def cell(x, y, dx=0.0, dy=0.0):
        return x + 0.5 + dx, H - y - 0.5 + dy

    if env_name == 'Hunt':
        xs, ys = int(coords[4]), int(coords[5])
        for i in range(6, len(coords) - 1, 2):
            ax.add_patch(patches.FancyBboxPatch(
                (coords[i] + 0.15, H - coords[i + 1] - 0.85), 0.7, 0.7,
                boxstyle="round,pad=0.05", color='#4caf50', zorder=2
            ))
        ax.add_patch(patches.Circle(cell(xs, ys), 0.3, color='#8B4513', zorder=3))
        ax.text(*cell(xs, ys), 'S', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=4)

    elif env_name == 'Harvest':
        for i in range(4, len(coords) - 2, 3):
            is_mature = bool(coords[i + 2])
            color = '#FF8C00' if is_mature else '#4caf50'
            ax.add_patch(patches.FancyBboxPatch(
                (coords[i] + 0.15, H - coords[i + 1] - 0.85), 0.7, 0.7,
                boxstyle="round,pad=0.05", color=color, zorder=2
            ))

    elif env_name == 'Escalation':
        xm, ym = int(coords[4]), int(coords[5])
        ax.add_patch(patches.RegularPolygon(
            cell(xm, ym), numVertices=4, radius=0.38,
            orientation=np.pi / 4, color='#FFC107', zorder=3
        ))
        ax.text(*cell(xm, ym), 'M', ha='center', va='center',
                fontsize=11, fontweight='bold', color='black', zorder=4)

    # Agents always offset so they stay visible even when on the same cell
    ax.add_patch(patches.Circle(cell(xa, ya, dx=-0.17), 0.27, color='#1565C0', zorder=5))
    ax.text(*cell(xa, ya, dx=-0.17), 'A', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=6)

    ax.add_patch(patches.Circle(cell(xb, yb, dx=+0.17), 0.27, color='#C62828', zorder=5))
    ax.text(*cell(xb, yb, dx=+0.17), 'B', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=6)

    legend_items = [
        patches.Patch(color='#1565C0', label='Agent A'),
        patches.Patch(color='#C62828', label='Agent B'),
    ]
    if env_name == 'Hunt':
        legend_items.append(patches.Patch(color='#4caf50', label='Plant'))
    if env_name == 'Harvest':
        legend_items.append(patches.Patch(color='#4caf50', label='Young plant'))
        legend_items.append(patches.Patch(color='#FF8C00', label='Mature plant'))
    if env_name == 'Hunt':
        legend_items.append(patches.Patch(color='#8B4513', label='Stag'))
    if env_name == 'Escalation':
        legend_items.append(patches.Patch(color='#FFC107', label='Mark'))

    ax.legend(handles=legend_items, loc='upper right', fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=AGENT_KEYS, required=True,
                        help='Agent type: a2c | ppo | mappo | random')
    parser.add_argument('--env', choices=ENVIRONMENTS, default='Hunt',
                        help='Environment: Hunt | Harvest | Escalation')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Seconds between steps (default 0.1)')
    parser.add_argument('--robust', action='store_true',
                        help='Load robust model')
    parser.add_argument('--adaptive', action='store_true',
                        help='Load adaptive model')
    parser.add_argument('--no-stag-follows', dest='stag_follows', action='store_false', default=True,
                        help='Stag moves randomly (Hunt only)')
    parser.add_argument('--social', action='store_true',
                        help='Load model trained with social (team) reward')
    parser.add_argument('--deterministic', action='store_true',
                        help='Argmax policy instead of sampling')
    parser.add_argument('--renderer', choices=['matplotlib', 'pygame'], default='matplotlib',
                        help='Renderer backend (default: matplotlib)')
    args = parser.parse_args()

    with open('configs/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    grid_size = tuple(config['env']['grid_size'])

    env_kwargs = dict(
        grid_size=grid_size,
        obs_type=config['env']['obs_type'],
        max_timesteps=config['env']['max_timesteps'],
        enable_multiagent=config['env']['enable_multiagent'],
        flip_obs=config['env']['flip_obs'],
        load_renderer=(args.renderer == 'pygame'),
    )
    if args.env == 'Hunt':
        env_kwargs['stag_follows'] = False if args.adaptive else args.stag_follows
    env = gym.make(f"StagHunt-{args.env}-v0", **env_kwargs)
    # pygame is now initialized (if requested) — safe to import TensorFlow

    from agents.a2c_agent import A2CAgent
    from agents.ppo_agent import PPOAgent
    from agents.mappo_agent import MAPPOAgent
    from agents.random_agent import RandomAgent
    AGENT_CLASSES = {
        'a2c': (A2CAgent, 'a2c'),
        'ppo': (PPOAgent, 'ppo'),
        'mappo': (MAPPOAgent, 'mappo'),
        'random': (RandomAgent, None),
    }

    obs, _ = env.reset()
    obs_dim = obs[0].shape[0]
    action_dim = env.action_space.n

    AgentClass, config_key = AGENT_CLASSES[args.agent]
    agent_config = config[config_key] if config_key else {}
    results_dir = config['logging']['results_dir']

    game_prefix = args.env.lower()
    if args.adaptive:
        model_name = f'{game_prefix}_agent_adaptive'
    elif args.robust:
        model_name = f'{game_prefix}_agent_robust'
    else:
        model_name = f'{game_prefix}_agent'

    if args.social:
        model_name += '_social'

    if args.agent == 'mappo':
        shared_agent = MAPPOAgent(obs_dim, action_dim, obs_dim, agent_config)
    else:
        shared_agent = AgentClass(obs_dim, action_dim, agent_config)
    if args.agent != 'random':
        shared_agent.load(f"{results_dir}/{model_name}_0")
    agents = [shared_agent, shared_agent]

    if args.renderer == 'matplotlib':
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        plt.show()

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            actions = []
            for i, agent in enumerate(agents):
                action, _, _ = agent.act(obs[i], deterministic=args.deterministic)
                actions.append(action)

            obs, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            total_reward += sum(rewards)
            step += 1

            if args.renderer == 'matplotlib':
                draw_state(ax, obs, grid_size, episode, step, total_reward, args.env)
                plt.pause(args.delay)
            else:
                env.render()
                time.sleep(args.delay)

        print(f"Episode {episode}/{args.episodes} — total reward: {total_reward:.1f}")

    if args.renderer == 'matplotlib':
        plt.ioff()
        plt.show()
    env.close()


if __name__ == "__main__":
    main()
