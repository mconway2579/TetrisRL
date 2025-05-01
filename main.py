from train_ppo import train_ppo
from train_dqn import train_dqn
from Enviorments import get_tetris_env, get_mcd_env, get_tetris_env_flat
import os

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    frames = 1_000_000
    # Train PPO
    # train_ppo(get_mcd_env, "mcd", total_frames=frames)
    #train_ppo(get_tetris_env, "tetris", total_frames=frames)
    train_ppo(get_tetris_env_flat, "tetris_flat", total_frames=frames)



    # Train DQN
    # train_dqn(get_mcd_env, "mcd", total_frames=frames)
   # train_dqn(get_tetris_env, "tetris", total_frames=frames)
    train_dqn(get_tetris_env_flat, "tetris_flat", total_frames=frames)

