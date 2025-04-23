import torch
from tensordict import TensorDict
from torchrl.envs import GymWrapper
from torchrl.envs.utils import RandomPolicy
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from TetrisEnv import get_env
from utils import select_device

    
import cv2

N_WORKERS = 8
FRAMES_PER_WORKER = 256
FRAMES_PER_COLLECTER = FRAMES_PER_WORKER*N_WORKERS
REPLAY_BUFFER_BATCH_SIZE = 32
BATCHES_TO_STORE = 1024
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
        device=device,
        ndim=1
    )
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=SamplerWithoutReplacement(),  # you can swap in PrioritizedSampler, etc.
        batch_size=REPLAY_BUFFER_BATCH_SIZE
    )
    return replay_buffer
if __name__ == "__main__":
    device = select_device()
    print(f"Using device: {device}")

    env = get_env()
    # Create a random policy
    policy = RandomPolicy(env.action_spec)
    eval_rollout = env.rollout(1000, policy)
    n_workers = 8
    fpb = 256
    collector = get_collecter(get_env, policy)
    replay_buffer = get_replay_buffer()

    for count, batch_td in enumerate(collector):
        print(f"Batch {count}: {batch_td.shape}")
        batch_size = batch_td.shape[0]
        replay_buffer.extend(batch_td)
        for j in range(batch_size):
            initial_state = batch_td["observation"][j]
            next_state = batch_td["next", "observation"][j]
            # done flags
            d = batch_td["done"][j]                    
            terminated = batch_td["terminated"][j]     
            truncated  = batch_td["truncated"][j]      # 
            #print(f"Initial state {j} {i}: {initial_state.shape} {d=}, {terminated=}, {truncated=}")
            initial_img_np = initial_state.squeeze(0).cpu().numpy()  # now shape [20, 10, 3]
            #next_img_np = next_state.squeeze(0).cpu().numpy()
            cv2.imshow(f"State", initial_img_np.transpose(1,2,0))
            #cv2.imshow(f"Next State {i}", next_img_np)
            cv2.waitKey(1)
        
            #input("press enter to see state pair")
        for k, v in batch_td.items():
            print(f"{k}: {v.shape}")
        print("\n\n")
        if count > 1:
            break
    print("Done collecting data")
    while True:
        train_td = replay_buffer.sample()
        batch_size = train_td.size()[0]
        for i in range(batch_size):
            initial_state = train_td["observation"][i]
            next_state = train_td["next", "observation"][i]
            cv2.imshow(f"ReplayBuffer State", initial_state.squeeze(0).cpu().numpy().transpose(1,2,0))
            cv2.waitKey(1)
            input("press enter to see next state")
            cv2.imshow(f"ReplayBuffer State", next_state.squeeze(0).cpu().numpy().transpose(1,2,0))
            cv2.waitKey(1)
            input("press enter to see next sample")

            print(f"Train batch: {train_td}")
        #print(f"")
    exit(0)
