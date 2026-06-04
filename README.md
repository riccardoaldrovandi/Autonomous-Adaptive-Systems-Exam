---

  # Multi-Agent Reinforcement Learning in Stag-Hunt Environments

  Exam project for the course **Autonomous and Adaptive Systems** (A.A.
  2025/2026) — University of Bologna.

  Three RL algorithms (A2C, PPO, MAPPO) trained as Independent Learners in the
  [Gymnasium 
  Stag-Hunt](https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt)
  multi-agent environment, across three cooperative games: **Hunt**,
  **Harvest**, and **Escalation**.

  ---

  ## Algorithms
  | Algorithm | Update frequency | Advantage |
  |-----------|-----------------|-----------|
  | A2C | Every T=5 steps | n-step return |
  | PPO | Every T=256 steps (rollout) | GAE (λ=0.95), K=4 epochs |
  | MAPPO | Every T=256 steps (rollout) | GAE + centralized critic |

  All agents share network weights between the two players via `flip_obs=True`
  (the second agent observes itself as agent 1).

  **Architecture**: Actor and Critic are separate networks — `Input → Dense(64,
  relu) → Dense(64, relu) → output`.
  ---

  ## Environments
```
  | Game | Cooperation structure | Key finding |
  | Hunt | Coordinate to catch stag | A2C learns passive exploit (stay still);
  PPO/MAPPO converge to hare equilibrium |
  | Harvest | Time cooperative harvest before plant dies | A2C >> PPO > MAPPO
  (timing credit assignment) |
  | Escalation | Maintain mutual cooperation streak | PPO >> MAPPO >> A2C
  (rollout horizon) |
  ```
  ### Training variants
  
  - **Standard**: fixed `stag_follows=True` (Hunt only)
  - **Robust**: stochastic `stag_follows` (domain randomization)
  - **Adaptive**: curriculum — train first with `stag_follows=True`, fine-tune
  with `stag_follows=False`
  
  ---
  
  ## Project structure
   ```

  .
  ├── agents/
  │   ├── base_agent.py       # abstract base class
  │   ├── a2c_agent.py
  │   ├── ppo_agent.py
  │   ├── mappo_agent.py
  │   └── random_agent.py
  ├── training/
  │   ├── trainer.py          # A2C / PPO training loop
  │   ├── mappo_trainer.py    # MAPPO training loop (centralized critic)
  │   └── evaluator.py
  ├── configs/
  │   └── default_config.yaml
  ├── train.py                # CLI entry point for training
  ├── evaluate.py             # CLI entry point for evaluation
  ├── main.py                 # live visualization (matplotlib)
  ├── train_evo.py            # evolutionary hyperparameter search (proof of concept)
  └── results/                # saved weights and training logs
 ```

  ---

  ## Installation

  ```bash
  # 1. Install Gymnasium Stag-Hunt from source
  git clone https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt.git
  cd Gymnasium-Stag-Hunt && pip install . && cd ..

  # 2. Install dependencies
  pip install -r requirements.txt

  Python 3.11, TensorFlow 2.16.2.

  ---
  Usage

  Train

  # Standard training
  python train.py --agent a2c --game Hunt

  # Robust training (stochastic stag_follows)
  python train.py --agent ppo --game Hunt --robust

  # Adaptive fine-tuning (loads standard weights, trains without stag_follows)
  python train.py --agent mappo --game Hunt --adaptive

  # Override number of episodes
  python train.py --agent a2c --game Harvest --episodes 500

  Supported values: --agent {a2c, ppo, mappo}, --game {Hunt, Harvest, 
  Escalation}.

  Evaluate

  python evaluate.py --agent a2c --game Hunt
  python evaluate.py --agent ppo --game Escalation
  python evaluate.py --agent mappo --game Hunt --adaptive

  Live demo (matplotlib visualization)

  # Hunt — A2C passive exploit
  python main.py --agent a2c --env Hunt --deterministic --delay 0.3

  # Harvest — cooperative harvesting
  python main.py --agent a2c --env Harvest --deterministic --delay 0.3

  # Escalation — PPO streak (no --deterministic: stochastic policy needed to 
  follow the moving mark)
  python main.py --agent ppo --env Escalation --delay 0.3

  # Hunt adaptive — hare equilibrium
  python main.py --agent mappo --env Hunt --adaptive --no-stag-follows
  --deterministic --delay 0.3

  ---
  Key results (100 evaluation episodes)

  Hunt (standard, stag_follows=True)

  ┌────────┬─────────────┬─────────────────┬───────┐
  │ Agent  │ Avg reward  │ Stag catches/ep │ Coop% │
  ├────────┼─────────────┼─────────────────┼───────┤
  │ A2C    │ 2394 ± 100  │ 240             │ 100%  │
  ├────────┼─────────────┼─────────────────┼───────┤
  │ PPO    │ 344 ± 3324  │ 174             │ 72%   │
  ├────────┼─────────────┼─────────────────┼───────┤
  │ MAPPO  │ −22 ± 248   │ 0.15            │ 6%    │
  ├────────┼─────────────┼─────────────────┼───────┤
  │ Random │ −1630 ± 105 │ 11              │ 100%  │
  └────────┴─────────────┴─────────────────┴───────┘

  Harvest (standard)

  ┌────────┬────────────┬──────────────────┬───────┐
  │ Agent  │ Avg reward │ Coop harvests/ep │ Coop% │
  ├────────┼────────────┼──────────────────┼───────┤
  │ A2C    │ 1743 ± 128 │ 178              │ 100%  │
  ├────────┼────────────┼──────────────────┼───────┤
  │ PPO    │ 815 ± 173  │ 107              │ 100%  │
  ├────────┼────────────┼──────────────────┼───────┤
  │ MAPPO  │ 654 ± 162  │ 90               │ 100%  │
  ├────────┼────────────┼──────────────────┼───────┤
  │ Random │ 357 ± 34   │ 68               │ 100%  │
  └────────┴────────────┴──────────────────┴───────┘

  Escalation (standard)

  ┌────────┬─────────────┬────────────────┬───────┐
  │ Agent  │ Avg reward  │ Mutual coop/ep │ Coop% │
  ├────────┼─────────────┼────────────────┼───────┤
  │ PPO    │ 67 ± 68     │ 33             │ 100%  │
  ├────────┼─────────────┼────────────────┼───────┤
  │ MAPPO  │ 11 ± 23     │ 5.3            │ 46%   │
  ├────────┼─────────────┼────────────────┼───────┤
  │ A2C    │ 0.26 ± 1.19 │ 0.13           │ 6%    │
  ├────────┼─────────────┼────────────────┼───────┤
  │ Random │ 2.33 ± 2.09 │ 1.3            │ 70%   │
  └────────┴─────────────┴────────────────┴───────┘

  ---
  Configuration

  All hyperparameters are in configs/default_config.yaml.

  ┌──────────────┬──────┬─────────────┐
  │    Param     │ A2C  │ PPO / MAPPO │
  ├──────────────┼──────┼─────────────┤
  │ lr           │ 1e-3 │ 3e-4        │
  ├──────────────┼──────┼─────────────┤
  │ γ            │ 0.99 │ 0.99        │
  ├──────────────┼──────┼─────────────┤
  │ T (steps)    │ 5    │ 256         │
  ├──────────────┼──────┼─────────────┤
  │ entropy coef │ 0.1  │ 0.01        │
  ├──────────────┼──────┼─────────────┤
  │ value coef   │ 0.5  │ 0.5         │
  ├──────────────┼──────┼─────────────┤
  │ GAE λ        │ —    │ 0.95        │
  ├──────────────┼──────┼─────────────┤
  │ clip ε       │ —    │ 0.2         │
  ├──────────────┼──────┼─────────────┤
  │ K epochs     │ —    │ 4           │
  └──────────────┴──────┴─────────────┘

  ---
  Author

  Riccardo Aldrovandi — riccardo.aldrovandi3@unibo.it — riccardo.aldro@gmail.com
  Course: Autonomous and Adaptive Systems
  University of Bologna, A.A. 2025/2026

  ---
