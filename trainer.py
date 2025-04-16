import torch
from matplotlib import pyplot as plt
from TetrisEnv import TetrisEnv
from torchrl.envs import GymWrapper  # import the GymWrapper from TorchRL



if __name__ == "__main__":
    # Create the environment
    gym_env = TetrisEnv()
    env = GymWrapper(gym_env)
    # Reset the environment to get the initial observation (converted to a tensor)
    observation = env.reset()
    print(f"{observation=}")

    # Take a random action from the action space
    step_outcome = env.rand_step()

    # Unpack the outcome
    next_observation, reward, done, info = step_outcome

    print("Initial Observation:", observation)
    print("Next Observation:", next_observation)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)