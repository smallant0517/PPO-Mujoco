import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from config import PPOConfig
from utils import set_seed, ensure_dir, explained_variance
from envs import make_vec_env, get_space_info
from model import ActorCritic
from buffer import RolloutBuffer

@torch.no_grad()
def quick_eval(model: ActorCritic, env_id: str, episodes: int, device: torch.device):
    import gymnasium as gym
    env = gym.make(env_id)
    model.eval()
    rets = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = model.get_dist(obs_t)
            action = dist.mean[0].cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep += float(r)
        rets.append(ep)
    env.close()
    model.train()
    return float(np.mean(rets))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.env_id is not None:
        cfg.env_id = args.env_id
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)

    envs = make_vec_env(cfg.env_id, cfg.seed, cfg.num_envs)
    obs_shape, act_dim, act_low, act_high = get_space_info(envs)
    obs_dim = obs_shape[0]

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # For action squashing: PPO 常見做法是「不tanh，直接依action_space裁切」
    act_low_t = torch.tensor(act_low, dtype=torch.float32, device=device)
    act_high_t = torch.tensor(act_high, dtype=torch.float32, device=device)

    # Init obs
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # Derived
    batch_size = cfg.num_envs * cfg.rollout_steps
    num_updates = cfg.total_timesteps // batch_size
    minibatches_per_epoch = batch_size // cfg.minibatch_size
    assert batch_size % cfg.minibatch_size == 0, "batch_size must be divisible by minibatch_size."

    global_step = 0
    start_time = time.time()

    pbar = trange(num_updates, desc="PPO Updates", leave=True)
    for update in pbar:
        # LR anneal
        if cfg.anneal_lr:
            frac = 1.0 - (update / float(num_updates))
            lrnow = frac * cfg.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lrnow

        buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, obs_shape, act_dim, device)

        # Collect rollout
        for t in range(cfg.rollout_steps):
            global_step += cfg.num_envs

            action, logprob, value, _ = model.act(obs)
            # clip to action bounds
            action_clipped = torch.max(torch.min(action, act_high_t), act_low_t)

            next_obs, reward, terminated, truncated, _ = envs.step(action_clipped.detach().cpu().numpy())
            done = np.logical_or(terminated, truncated).astype(np.float32)

            reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
            done_t = torch.tensor(done, dtype=torch.float32, device=device)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            buffer.add(obs, action, logprob, reward_t, done_t, value)

            obs = next_obs_t

        # Bootstrap value
        with torch.no_grad():
            next_value = model.get_value(obs)  # [N]
        buffer.compute_gae(next_value, cfg.gamma, cfg.gae_lambda)

        b_obs, b_actions, b_logprobs, b_adv, b_ret, b_val = buffer.get()

        # Advantage normalize
        if cfg.normalize_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # PPO update
        inds = np.arange(batch_size)
        clipfracs = []
        approx_kls = []
        pg_losses = []
        v_losses = []
        ent_losses = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb_inds = inds[start:start + cfg.minibatch_size]

                newlogprob, entropy, newvalue = model.evaluate_actions(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = (ratio - 1 - logratio).mean()
                    approx_kls.append(approx_kl.item())
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    clipfracs.append(clipfrac)

                # Policy loss
                mb_adv = b_adv[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped)
                mb_ret = b_ret[mb_inds]
                if cfg.clip_vloss:
                    v_unclipped = (newvalue - mb_ret) ** 2
                    v_clipped = b_val[mb_inds] + torch.clamp(
                        newvalue - b_val[mb_inds],
                        -cfg.value_clip_coef, cfg.value_clip_coef
                    )
                    v_clipped_loss = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_ret) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

            # Early stop by KL
            if cfg.target_kl is not None and len(approx_kls) > 0:
                if np.mean(approx_kls[-minibatches_per_epoch:]) > cfg.target_kl:
                    break

        # Logging
        y_pred = b_val
        y_true = b_ret
        ev = explained_variance(y_pred, y_true)

        sps = int(global_step / (time.time() - start_time))
        info = (
            f"upd={update+1}/{num_updates} "
            f"step={global_step} "
            f"pg={np.mean(pg_losses):.4f} "
            f"v={np.mean(v_losses):.4f} "
            f"ent={np.mean(ent_losses):.4f} "
            f"kl={np.mean(approx_kls):.4f} "
            f"clipfrac={np.mean(clipfracs):.3f} "
            f"ev={ev:.3f} "
            f"sps={sps}"
        )
        pbar.set_postfix_str(info)

        # Save
        if (update + 1) % cfg.save_every_updates == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"ppo_{cfg.env_id}_upd{update+1}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # Eval
        if (update + 1) % cfg.eval_interval_updates == 0:
            avg_ret = quick_eval(model, cfg.env_id, cfg.eval_episodes, device)
            print(f"\n[Eval] update={update+1} avg_return={avg_ret:.2f}\n")

    # Final save
    final_path = os.path.join(cfg.save_dir, f"ppo_{cfg.env_id}_final.pt")
    torch.save(model.state_dict(), final_path)
    envs.close()
    print(f"Done. Final checkpoint: {final_path}")

if __name__ == "__main__":
    main()