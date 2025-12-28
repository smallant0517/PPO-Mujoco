import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()

        # Policy network
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        # log_std as parameter (state-independent)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Value network
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Init (common PPO style)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.pi[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.v[-1].weight, gain=1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mu = self.pi(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def act(self, obs: torch.Tensor):
        dist = self.get_dist(obs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        value = self.get_value(obs)
        entropy = dist.entropy().sum(-1)
        return action, logprob, value, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.get_dist(obs)
        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return logprob, entropy, value
