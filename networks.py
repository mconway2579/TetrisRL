import torch
import torch.nn as nn
from Enviorments import get_tetris_env, get_mcd_env, get_tetris_env_flat
from utils import select_device
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor, TanhNormal
from torch.distributions import Categorical
from tensordict.nn.distributions import NormalParamExtractor

import gym
import gymnasium 
import cv2
from torchrl.modules import EGreedyModule, QValueActor

device = select_device()

BoxTypes   = (gym.spaces.Box,   gymnasium.spaces.Box)
DiscTypes  = (gym.spaces.Discrete, gymnasium.spaces.Discrete)

hidden_dim = 512

class ActionNetwork(nn.Module):
    def __init__(self, get_env_func):
        super().__init__()
        env = get_env_func()
        
        obs_space = env.observation_space["state"]
        #print(f"{obs_space.shape=}")
        first_layer = None
        if isinstance(obs_space, BoxTypes) and len(obs_space.shape) == 3:
            first_layer = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
                nn.Flatten(start_dim=1),
            )
        elif len(obs_space.shape) == 1:
            first_layer = nn.Sequential(
                nn.Linear(obs_space.shape[0], hidden_dim), nn.ReLU()
            )
        else:
            raise ValueError(f"Unsupported observation space shape: {obs_space}")
        

        observation = env.reset()["observation"]
        dummy_input = torch.zeros(1, *observation.shape)
        dummy_output = first_layer(dummy_input)
        flattened_shape = dummy_output.shape[1]

        self.actor = nn.Sequential(
            first_layer,
            nn.Linear(flattened_shape, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, env.action_space.n)
        )
        with torch.no_grad():  # avoid tracking in autograd
            for layer in self.actor:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        logits = self.actor(obs)
        return logits
    
class ValueEstimator(nn.Module):
    def __init__(self, get_env_func):
        super().__init__()
        env = get_env_func()
        obs_space = env.observation_space["state"]
        #print(f"{obs_space.shape=}")
        first_layer = None
        if isinstance(obs_space, BoxTypes) and len(obs_space.shape) == 3:
            first_layer = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
                nn.Flatten(start_dim=1),
            )
        elif len(obs_space.shape) == 1:
            first_layer = nn.Sequential(
                nn.Linear(obs_space.shape[0], hidden_dim), nn.ReLU()
            )
        else:
            raise ValueError(f"Unsupported observation space shape: {obs_space}")
        observation = env.reset()["observation"]
        #print(f"observation shape: {observation.shape}")
        dummy_input = torch.zeros(1, *observation.shape)
        dummy_output = first_layer(dummy_input)
        flattened_shape = dummy_output.shape[1]
        #print(f"dummy_input shape: {dummy_input.shape}")
        #print(f"dummy_output shape: {dummy_output.shape}")
        self.critic  = nn.Sequential(
            first_layer,
            nn.Linear(flattened_shape, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        with torch.no_grad():           # avoid tracking in autograd
            for layer in self.critic:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
           

    def forward(self, obs: torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        value  = self.critic(obs)
        return value

def get_PPO_policy(get_env_func):
    base_actor = ActionNetwork(get_env_func)                      # keeps forward(obs) if you like
    print(f"{ActionNetwork=}")
    base_value = ValueEstimator(get_env_func)
    
    value_module = TensorDictModule(
        module   = base_value,
        in_keys  = ["observation"],             # what the env produces
        out_keys = ["state_value"]  # what you want stored
    ).to(device)
    env = get_env_func()
    action_space = env.action_space
    actor = None
    if isinstance(action_space, DiscTypes):
        ppo_p    = TensorDictModule(
            module   = base_actor,
            in_keys  = ["observation"],             # what the env produces
            out_keys = ["logits"]  # what you want stored
        ).to(device)
        actor = ProbabilisticActor(
            module=ppo_p,
            spec = env.action_spec,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class = Categorical,
            return_log_prob    = True   # <-- writes the log‐prob to the tensordict
        ).to(device)
    elif isinstance(action_space, BoxTypes):
        ppo_p    = TensorDictModule(
            module   = base_actor,
            in_keys  = ["observation"],             # what the env produces
            out_keys = ["loc", "scale"]  # what you want stored
        ).to(device)
        actor = ProbabilisticActor(
            module=ppo_p,
            spec = env.action_spec,
            in_keys=["loc", "scale"],        # <-- Continuous → needs mean/std
            out_keys=["action"],
            distribution_class=TanhNormal,   # TanhNormal for bounded continuous
            distribution_kwargs={
                "low": env.action_spec_unbatched.space.low,
                "high": env.action_spec_unbatched.space.high,
            },
            return_log_prob=True
        ).to(device)
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")
    #print(f"{actor=}")
    #input("Press enter to continue")
    return actor, value_module


#https://pytorch.org/rl/main/tutorials/coding_dqn.html
def get_EGDQN(get_env_func, eps_start, eps_end, total_frames):
    env = get_env_func()
    actor_net = ActionNetwork(get_env_func)
    actor = QValueActor(actor_net, in_keys=["observation"], spec = env.action_spec).to(device)
    exploration_module = EGreedyModule(
        spec = env.action_spec,
        annealing_num_steps=total_frames,
        eps_init=eps_start,
        eps_end=eps_end
    )
    actor_explore = TensorDictSequential(
        actor,
        exploration_module
    ).to(device)
    return actor, actor_explore


if __name__ == "__main__":
    from data_collector import get_collecter, get_replay_buffer
    get_env_func = get_tetris_env
    #get_env_func = get_mcd_env
    #get_env_func = get_tetris_env_flat


    predictor1, predictor2 = get_PPO_policy(get_env_func)
    #https://pytorch.org/rl/main/tutorials/coding_dqn.html

    #print(f"{ppo_policy=}")
    env = get_env_func()
    print("Running policy:", predictor1(env.reset()))
    print("Running value:", predictor2(env.reset()))
    #input("Press enter to continue")
    #collector = get_collecter(get_tetris_env, ppo_policy)
    collector = get_collecter(get_env_func, predictor1, 256, 100_000)
    replay_buffer = get_replay_buffer(1024, 256, 32)

    for count, batch_td in enumerate(collector):
        print(f"Batch {count}: {batch_td.shape}")
        batch_size = batch_td.shape[0]
        print(f"Batch: {batch_td}")
        replay_buffer.extend(batch_td)
        for j in range(batch_size):
            initial_img = batch_td["pixels"][j].squeeze(0).permute(1,2,0).cpu().numpy()
            next_img = batch_td["next", "pixels"][j].squeeze(0).permute(1,2,0).cpu().numpy()
            # done flags
            d = batch_td["done"][j]                    
            terminated = batch_td["terminated"][j]     
            truncated  = batch_td["truncated"][j]      # 
            #print(f"Initial state {j} {i}: {initial_state.shape} {d=}, {terminated=}, {truncated=}")
            #next_img_np = next_state.squeeze(0).cpu().numpy()
            cv2.imshow(f"State", initial_img)
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
        train_td = replay_buffer.sample().to(device)
        batch_size = train_td.size()[0]
        out_p = predictor1(train_td)
        out_v = predictor2(train_td)
        print(f"out_v: {out_v}")
        print(f"out_p: {out_p}")
        for i in range(batch_size):
            initial_img = train_td["pixels"][i].squeeze(0).cpu().numpy().transpose(1,2,0)
            next_img = train_td["next", "pixels"][i].squeeze(0).cpu().numpy().transpose(1,2,0)
            cv2.imshow(f"ReplayBuffer State", initial_img)
            cv2.waitKey(1)
            input("press enter to see next state")
            cv2.imshow(f"ReplayBuffer State", next_img)
            cv2.waitKey(1)
            input("press enter to see next sample")

            #print(f"Train batch: {train_td}")
