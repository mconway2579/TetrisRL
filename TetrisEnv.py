import gym
import cv2
import numpy as np
# This will work only if you have a properly configured Tetris ROM and environment.
import torch.nn.functional as F




class TetrisEnv(gym.Env):
    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.env = gym.make('ALE/Tetris-v5', render_mode='rgb_array', repeat_action_probability=0, frameskip = 4)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.last_observation = None

    def reset(self):
        self.env.reset()
        return self.step(0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs[27:203, 22:64]
        #obs[obs==111] = 0
        #obs[obs!=0] = 255
        for i in range(obs.shape[0]):
            if (i+0) % 8 == 0:
                    obs[i,:, 1] = 255
        for j in range(obs.shape[1]):
            if (j-1) % 4 == 0:
                obs[:, j, 2] = 255

        #obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        
        
        self.last_observation = obs
        return obs, reward, terminated, truncated, info 

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
    

if __name__ == "__main__":
    env = TetrisEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your policy or agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, {type(action)}")
        print(f"Observation: {obs.shape}, {type(obs)}")
        cv2.imshow("Tetris", obs)
        cv2.waitKey(1)
        done = terminated or truncated
    cv2.waitKey(0)
    env.close()
