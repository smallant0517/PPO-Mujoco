from dataclasses import dataclass

@dataclass
class PPOConfig:
    # Env
    env_id: str = "Hopper-v5"
    seed: int = 42

    # Training steps
    total_timesteps: int = 2_000_000
    num_envs: int = 8
    rollout_steps: int = 2048  # T
    # total batch size = num_envs * rollout_steps
    update_epochs: int = 10
    minibatch_size: int = 256

    # PPO hyperparams
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03  # optional early stop

    # Optim
    learning_rate: float = 3e-4
    anneal_lr: bool = True

    # Tricks
    normalize_adv: bool = True
    clip_vloss: bool = True
    value_clip_coef: float = 0.2  # only used if clip_vloss=True

    # Logging / saving
    log_interval: int = 10
    save_dir: str = "checkpoints"
    save_every_updates: int = 20

    # Eval
    eval_episodes: int = 5
    eval_interval_updates: int = 20