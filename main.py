from train_ppo import train_ppo
from train_dqn import train_dqn
from Enviorments import get_tetris_env, get_mcd_env, get_tetris_env_flat
import os

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    total_frames = 1_048_576 * 4
    frames_per_collector = 512
    batches_to_store = 4096
    lr = 1e-4
    minibatch_size = 32
    training_iter_per_batch = 5
    # Train PPO
    # train_ppo(get_mcd_env, "mcd", total_frames=frames)
    train_ppo(get_tetris_env, "tetris", total_frames=total_frames, 
              frames_per_collector=frames_per_collector,
              batches_to_store=batches_to_store,
              lr=lr,
              mini_batch_size=minibatch_size,
              training_iter_per_batch=training_iter_per_batch)
    train_ppo(get_tetris_env_flat, "tetris_flat", total_frames=total_frames, 
              frames_per_collector=frames_per_collector,
              batches_to_store=batches_to_store,
              lr=lr,
              mini_batch_size=minibatch_size,
              training_iter_per_batch=training_iter_per_batch)


    # Train DQN
    # train_dqn(get_mcd_env, "mcd", total_frames=frames)
    train_dqn(get_tetris_env, "tetris", total_frames=total_frames, 
              frames_per_collector=frames_per_collector,
              batches_to_store=batches_to_store,
              lr=lr,
              mini_batch_size=minibatch_size,
              training_iter_per_batch=training_iter_per_batch)
    train_dqn(get_tetris_env_flat, "tetris_flat", total_frames=total_frames, 
              frames_per_collector=frames_per_collector,
              batches_to_store=batches_to_store,
              lr=lr,
              mini_batch_size=minibatch_size,
              training_iter_per_batch=training_iter_per_batch)

