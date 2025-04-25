import torch
from tensordict import TensorDict
from torchrl.envs import GymWrapper
from torchrl.envs.utils import RandomPolicy
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from Enviorments import get_tetris_env, get_mc_env
from utils import select_device

import cv2

FRAMES_PER_COLLECTER = 256
REPLAY_BUFFER_BATCH_SIZE = 32
BATCHES_TO_STORE = 1024
TOTAL_FRAMES = 1_000_000
device = select_device()
def get_collecter(env_func, policy):
    """collector = MultiSyncDataCollector(
        create_env_fn=[env_func] * N_WORKERS,
        policy=policy,
        frames_per_batch=FRAMES_PER_COLLECTER,
        device = device,
        reset_at_each_iter=True,    # resets _between_ batches
        reset_when_done=True        # torn down _within_ a batch on done
    )"""
    collector = SyncDataCollector(
        create_env_fn=env_func,
        policy=policy,
        frames_per_batch=FRAMES_PER_COLLECTER,
        total_frames=TOTAL_FRAMES,
        device = device,
        reset_at_each_iter=True,    # resets _between_ batches
        reset_when_done=True        # torn down _within_ a batch on done
    )
    return collector
def get_replay_buffer():
    max_episodes = BATCHES_TO_STORE
    elements_per_batch = FRAMES_PER_COLLECTER
    storage_capacity = max_episodes * elements_per_batch
    storage = LazyTensorStorage(
        max_size=storage_capacity,
        device="cpu",
        ndim=1
    )
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=SamplerWithoutReplacement(),  # you can swap in PrioritizedSampler, etc.
        batch_size=REPLAY_BUFFER_BATCH_SIZE
    )
    return replay_buffer



if __name__ == "__main__":
    #print(list(GymEnv.available_envs))

    device = select_device()
    print(f"Using device: {device}")
    get_env_func = get_tetris_env
    #get_env_func = get_mc_env
    env = get_env_func()
    # Create a random policy
    policy = RandomPolicy(env.action_spec)
    eval_rollout = env.rollout(1000, policy)
    #collector = get_collecter(get_env, policy)
    collector = get_collecter(get_env_func, policy)

    replay_buffer = get_replay_buffer()

    for count, batch_td in enumerate(collector):
        print(f"Batch {count}: {batch_td.shape}")
        print(f"Batch: {batch_td}")
        batch_size = batch_td.shape[0]
        replay_buffer.extend(batch_td)
        for j in range(batch_size):
            # done flags
            d = batch_td["done"][j]                    
            terminated = batch_td["terminated"][j]     
            truncated  = batch_td["truncated"][j]      # 
            #print(f"Initial state {j} {i}: {initial_state.shape} {d=}, {terminated=}, {truncated=}")
            img_np = batch_td["pixels"][j].squeeze(0).permute(1,2,0).cpu().numpy()
            #next_img_np = next_state.squeeze(0).cpu().numpy()
            cv2.imshow(f"State", img_np)
            #cv2.imshow(f"Next State {i}", next_img_np)
            cv2.waitKey(1)
        
            #input("press enter to see state pair")
        for k, v in batch_td.items():
            print(f"{k}: {v.shape}")
        print("\n\n")
        if count > 1:
            break
    print("Done collecting data")
    count = 0
    while True:
        train_td = replay_buffer.sample()
        batch_size = train_td.size()[0]
        for i in range(batch_size):
            img = train_td["pixels"][i].permute(1,2,0).cpu().numpy()
            next_img = train_td["next", "pixels"][i].permute(1,2,0).cpu().numpy()
            cv2.imshow(f"ReplayBuffer State", img)
            cv2.waitKey(1)
            input("press enter to see next state")
            cv2.imshow(f"ReplayBuffer State", next_img)
            cv2.waitKey(1)
            input("press enter to see next sample")

            print(f"Train batch: {train_td}")
        count += 1
        if count >= 1:
            break
    
    


