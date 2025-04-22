# test_data_collector.py
import torch
from tensordict import TensorDict
from torchrl.envs import GymWrapper
from torchrl.envs.utils import RandomPolicy
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from TetrisEnv import get_env
from utils import select_device

    
import cv2
if __name__ == "__main__":
    device = select_device()
    print(f"Using device: {device}")

    env = get_env()
    # Create a random policy
    policy = RandomPolicy(env.action_spec)
    n_workers = 8
    fpb = 256
    collector = MultiSyncDataCollector(
        create_env_fn=[get_env] * n_workers,
        policy=policy,
        frames_per_batch=fpb * n_workers,
        device = device,
        reset_at_each_iter=True,    # resets _between_ batches
        reset_when_done=True        # torn down _within_ a batch on done
    )

    T = (fpb * n_workers) // n_workers  # == fpb
    N_batches = 1024
    max_episodes = N_batches
    elements_per_batch = n_workers * T
    storage_capacity = max_episodes * elements_per_batch


    storage = LazyTensorStorage(
        max_size=storage_capacity,
        device=device,
        ndim=2  # because each “element” is really a [n_workers, T]-shaped chunk
    )
    
    replay_buffer = ReplayBuffer(
        storage=storage,
        sampler=SamplerWithoutReplacement(),  # you can swap in PrioritizedSampler, etc.
        batch_size=4*T,                    # when sampling, get 4 trajectories of length T
    )


    for count, batch_td in enumerate(collector):
        #print(f"Batch {i}: {batch_td.shape}")
        worker_count = batch_td.batch_size[0]
        batch_size = batch_td.batch_size[1]
        replay_buffer.extend(batch_td)
        for j in range(batch_size):
            for i in range(worker_count):
                initial_state = batch_td["observation"][i][j]
                next_state = batch_td["next", "observation"][i][j]  # [n_workers, T, 1, 20, 10, 3]
                # done flags
                d = batch_td["done"][i][j]                    # [n_workers, T, 1]
                terminated = batch_td["terminated"][i][j]     # [n_workers, T, 1]
                truncated  = batch_td["truncated"][i][j]      # [n_workers, T, 1]
                #print(f"Initial state {j} {i}: {initial_state.shape} {d=}, {terminated=}, {truncated=}")
                initial_img_np = initial_state.squeeze(0).cpu().numpy()  # now shape [20, 10, 3]
                #next_img_np = next_state.squeeze(0).cpu().numpy()
                cv2.imshow(f"State {i}", initial_img_np)
                #cv2.imshow(f"Next State {i}", next_img_np)
            cv2.waitKey(1)
        
            #input("press enter to see state pair")
        #for k, v in batch_td.items():
        #    print(f"{k}: {v.shape}")
        if count > 1:
            break
    print("Done collecting data")
    while True:
        train_td = replay_buffer.sample()
        batch_size = train_td.size()[0]
        for i in range(batch_size):
            initial_state = train_td["observation"][i]
            next_state = train_td["next", "observation"][i]
            cv2.imshow(f"ReplayBuffer State", initial_state.squeeze(0).cpu().numpy())
            cv2.waitKey(1)
            input("press enter to see next state")
            cv2.imshow(f"ReplayBuffer State", next_state.squeeze(0).cpu().numpy())
            cv2.waitKey(1)
            input("press enter to see next sample")

            print(f"Train batch: {train_td}")
        #print(f"")
    exit(0)
