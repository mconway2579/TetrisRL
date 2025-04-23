import torch
import torch.nn as nn
import numpy as np
from TetrisEnv import get_env
from utils import select_device
from torchrl.collectors import MultiSyncDataCollector
from tensordict.nn import TensorDictModule

# constants
INPUT_SHAPE = (3, 20, 10)  # (C, H, W)
N_ACTIONS   = 4

# unpack and compute conv output size
C, H, W = INPUT_SHAPE
# conv1: (H−3)//2+1 → (20−3)//2+1 = 10
# conv2: (10−3)//2+1 = 5    → H_feat=5
# same for W: (10−3)//2+1 = 4, then conv2 → (4−3)//2+1 = 2 → W_feat=2
H_feat = ((H - 3)//2 + 1 - 3)//2 + 1  # = 5
W_feat = ((W - 3)//2 + 1 - 3)//2 + 1  # = 2
FLAT_FEATURE_SIZE = 64 * H_feat * W_feat  # = 64 * 5 * 2 = 640

class DQNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(FLAT_FEATURE_SIZE, 256), nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, obs: torch.Tensor):
        # expects obs shape [B, C, H, W]
        return self.net(obs)

class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(C, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(FLAT_FEATURE_SIZE, 256), nn.ReLU(),
        )
        self.actor  = nn.Linear(256, N_ACTIONS)
        self.critic = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor):
        f      = self.feat(obs)
        logits = self.actor(f)
        value  = self.critic(f).squeeze(-1)
        dist   = torch.distributions.Categorical(logits=logits)
        a      = dist.sample()
        return a, dist.log_prob(a), value

if __name__ == "__main__":
    device = select_device()
    print(f"Using device: {device}")

    dqn = DQNPolicy().to(device)
    ppo = PPOPolicy().to(device)

    env = get_env()
    reset_out = env.reset()
    # assume reset_out is a dict/tensordict with key "observation"
    obs = reset_out["observation"]  # this should be a Tensor of shape [C, H, W]

    # add batch dim and move to device
    obs_tensor = obs.unsqueeze(0).to(device)  # now [1, C, H, W]
    print("obs_tensor:", obs_tensor.shape)

    # DQN forward
    q = dqn(obs_tensor)
    a_dqn = q.argmax(dim=-1)

    # PPO forward
    a_ppo, logp, v = ppo(obs_tensor)

    print("DQN Q:", q)
    print("DQN action:", a_dqn)
    print("PPO action:", a_ppo)
    print("PPO log-prob:", logp)
    print("PPO value:", v)

    base_ppo = PPOPolicy()                      # keeps forward(obs) if you like
    ppo_td    = TensorDictModule(
        module   = base_ppo,
        in_keys  = ["observation"],             # what the env produces
        out_keys = ["action", "log_prob", "value"]  # what you want stored
    ).to(device)
    n_workers = 2
    fpb = 256
    ppo_collector = MultiSyncDataCollector(
        create_env_fn    = [get_env] * n_workers,
        policy           = ppo_td,              # <- use wrapped version
        frames_per_batch = fpb * n_workers,
        device           = device,
        reset_at_each_iter=True,
        reset_when_done=True,
    )
