import torch
import torch.nn as nn
from TetrisEnv import get_env
from utils import select_device
from tensordict.nn import TensorDictModule
import cv2
device = select_device()
#print(f"Using device: {device}")
# constants
INPUT_SHAPE = (3, 20, 10)  # (C, H, W)
N_ACTIONS   = 4

# unpack and compute conv output size
C, H, W = INPUT_SHAPE

class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        dummy_input = torch.zeros(1, *INPUT_SHAPE)
        dummy_output = self.conv(dummy_input)
        flattened_shape = dummy_output.shape[1]

        self.actor  = nn.Linear(flattened_shape, N_ACTIONS)

    def forward(self, obs: torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        f      = self.conv(obs)
        logits = self.actor(f)
        dist   = torch.distributions.Categorical(logits=logits)
        a      = dist.sample()
        return a, logits
    
class ValueEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        dummy_input = torch.zeros(1, *INPUT_SHAPE)
        dummy_output = self.conv(dummy_input)
        flattened_shape = dummy_output.shape[1]
        #print(f"dummy_input shape: {dummy_input.shape}")
        #print(f"dummy_output shape: {dummy_output.shape}")
        self.critic  = nn.Linear(flattened_shape, 1)

    def forward(self, obs: torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        f      = self.conv(obs)
        value  = self.critic(f).squeeze(-1)
        return value

def get_PPO_policy():
    base_ppo = PPOPolicy()                      # keeps forward(obs) if you like
    base_value = ValueEstimator()
    ppo_p    = TensorDictModule(
        module   = base_ppo,
        in_keys  = ["observation"],             # what the env produces
        out_keys = ["action", "logits"]  # what you want stored
    ).to(device)
    value_module = TensorDictModule(
        module   = base_value,
        in_keys  = ["observation"],             # what the env produces
        out_keys = ["value"]  # what you want stored
    ).to(device)
    
    return ppo_p, value_module

if __name__ == "__main__":
    from data_collector import get_collecter, get_replay_buffer
    ppo_policy, value_module = get_PPO_policy()
    print(f"{ppo_policy=}")    
    collector = get_collecter(get_env, ppo_policy)
    replay_buffer = get_replay_buffer()

    for count, batch_td in enumerate(collector):
        worker_count = batch_td.batch_size[0]
        batch_size = batch_td.batch_size[1]
        replay_buffer.extend(batch_td)
        for j in range(batch_size):
            for i in range(worker_count):
                initial_state = batch_td["observation"][i][j]
                next_state = batch_td["next", "observation"][i][j]
                # done flags
                d = batch_td["done"][i][j]                    
                terminated = batch_td["terminated"][i][j]     
                truncated  = batch_td["truncated"][i][j]      # 
                #print(f"Initial state {j} {i}: {initial_state.shape} {d=}, {terminated=}, {truncated=}")
                initial_img_np = initial_state.squeeze(0).cpu().numpy()  # now shape [20, 10, 3]
                #next_img_np = next_state.squeeze(0).cpu().numpy()
                cv2.imshow(f"State {i}", initial_img_np.transpose(1,2,0))
                #cv2.imshow(f"Next State {i}", next_img_np)
            cv2.waitKey(1)
        
            #input("press enter to see state pair")
        #for k, v in batch_td.items():
        #    print(f"{k}: {v.shape}")
        #print("\n\n")
        if count > 10:
           break
    print("Done collecting data")
    while True:
        train_td = replay_buffer.sample()
        batch_size = train_td.size()[0]
        out_p = ppo_policy(train_td)
        out_v = value_module(train_td)
        print(f"out_v: {out_v}")
        print(f"out_p: {out_p}")
        for i in range(batch_size):
            initial_state = train_td["observation"][i]
            next_state = train_td["next", "observation"][i]
            cv2.imshow(f"ReplayBuffer State", initial_state.squeeze(0).cpu().numpy().transpose(1,2,0))
            cv2.waitKey(1)
            input("press enter to see next state")
            cv2.imshow(f"ReplayBuffer State", next_state.squeeze(0).cpu().numpy().transpose(1,2,0))
            cv2.waitKey(1)
            input("press enter to see next sample")

            #print(f"Train batch: {train_td}")
