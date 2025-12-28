import torch

class RolloutBuffer:
    def __init__(self, T: int, N: int, obs_shape, act_dim: int, device: torch.device):
        self.T, self.N = T, N
        self.device = device

        self.obs = torch.zeros((T, N) + obs_shape, device=device)
        self.actions = torch.zeros((T, N, act_dim), device=device)
        self.logprobs = torch.zeros((T, N), device=device)
        self.rewards = torch.zeros((T, N), device=device)
        self.dones = torch.zeros((T, N), device=device)
        self.values = torch.zeros((T, N), device=device)

        self.advantages = torch.zeros((T, N), device=device)
        self.returns = torch.zeros((T, N), device=device)

        self._t = 0

    def add(self, obs, actions, logprobs, rewards, dones, values):
        t = self._t
        self.obs[t] = obs
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self._t += 1

    @torch.no_grad()
    def compute_gae(self, next_value, gamma: float, gae_lambda: float):
        # next_value: [N]
        T = self.T
        last_gae = torch.zeros((self.N,), device=self.device)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - self.dones[t]
            next_values = next_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get(self):
        # flatten: [T*N, ...]
        T, N = self.T, self.N
        b_obs = self.obs.reshape((T * N,) + self.obs.shape[2:])
        b_actions = self.actions.reshape((T * N,) + self.actions.shape[2:])
        b_logprobs = self.logprobs.reshape(T * N)
        b_advantages = self.advantages.reshape(T * N)
        b_returns = self.returns.reshape(T * N)
        b_values = self.values.reshape(T * N)
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values