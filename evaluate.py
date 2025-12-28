import argparse
import torch
import gymnasium as gym
from model import ActorCritic

@torch.no_grad()
def evaluate(env_id: str, ckpt_path: str, episodes: int, device: str):
    device = torch.device(device)
    env = gym.make(env_id)
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = ActorCritic(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = model.get_dist(obs_t)
            action = dist.mean[0].cpu().numpy()  # deterministic: mean action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        scores.append(ep_ret)

    env.close()
    avg = sum(scores) / len(scores)
    print(f"Env={env_id} episodes={episodes} avg_return={avg:.2f}")
    for i, s in enumerate(scores, 1):
        print(f"  ep{i}: {s:.2f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="Hopper-v5")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    evaluate(args.env_id, args.ckpt, args.episodes, args.device)

if __name__ == "__main__":
    main()