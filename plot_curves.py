import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

COLORS = {'a2c': '#1565C0', 'ppo': '#C62828', 'mappo': '#2E7D32', 'random': '#757575'}
LABELS = {'a2c': 'A2C', 'ppo': 'PPO', 'mappo': 'MAPPO', 'random': 'Random'}


def smooth(x, window=100):
    smoothed = np.array([
        np.mean(x[max(0, i - window // 2): i + window // 2 + 1])
        for i in range(len(x))
    ])
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Hunt', choices=['Hunt', 'Harvest', 'Escalation'])
    parser.add_argument('--agents', nargs='+', default=['a2c', 'ppo'],
                        choices=['a2c', 'ppo', 'mappo', 'random'])
    parser.add_argument('--window', type=int, default=100,
                        help='Smoothing window (episodes)')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out', default=None,
                        help='Output path (e.g. figures/hunt_curves.pdf)')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    game_prefix = args.game.lower()

    for agent_key in args.agents:
        path = f"{args.results_dir}/{game_prefix}_agent_{agent_key}_rewards.npy"
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        rewards = np.load(path)
        smoothed = smooth(rewards, args.window)
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, smoothed,
                color=COLORS[agent_key], label=LABELS[agent_key], linewidth=1.5)
        # faint raw signal
        ax.plot(episodes, rewards,
                color=COLORS[agent_key], alpha=0.15, linewidth=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward (sum of both agents)')
    ax.set_title(f'{args.game} — Training curves (window={args.window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        fig.savefig(args.out, dpi=200, bbox_inches='tight')
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
