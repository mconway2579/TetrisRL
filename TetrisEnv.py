import gym
import cv2
import numpy as np
# This will work only if you have a properly configured Tetris ROM and environment.
import torch.nn.functional as F


def process_default_observation(self, obs):
    #TODO
    #Use conv instead of nested loop

    obs = obs[27:203, 24:63]
    #cv2.imshow("old_obs", obs)
    #obs = cv2.resize(obs, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs[obs==111] = 0
    #obs[obs!=0] = 255
    #print(obs)
    row_centers = list(range(4, obs.shape[0], 8))
    col_centers = list(range(1, obs.shape[1], 4))

    new_obs = np.zeros((len(row_centers) , len(col_centers)), dtype=obs.dtype)
    for ni, i in enumerate(row_centers):
        for nj, j in enumerate(col_centers):
            #obs[i,j] = [255, 0, 255]
            new_obs[ni, nj] = 255 if obs[i,j] != 0 else 0
    return new_obs


class TetrisEnv(gym.Env):
    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.env = gym.make('ALE/Tetris-v5', render_mode='rgb_array', repeat_action_probability=0, frameskip = 8)
        self.action_space = self.env.action_space
        #self.observation_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(22, 10, 3), dtype=np.uint8)
        #print(self.observation_space)
        self.obs_t1 = None
        self.obs_t2 = None
        self.obs_t3 = None

    def reset(self):
        self.env.reset()
        obs_t1, reward, terminated, truncated, info = self.env.step(0)
        obs_t2, reward, terminated, truncated, info = self.env.step(0)
        obs_t3, reward, terminated, truncated, info = self.env.step(0)
        self.obs_t1 = process_default_observation(self, obs_t1)
        self.obs_t2 = process_default_observation(self, obs_t2)
        self.obs_t3 = process_default_observation(self, obs_t3)
        return self.get_observation(), info

    def get_observation(self):
        stacked_image = np.stack([self.obs_t1, self.obs_t2, self.obs_t3], axis=-1)
        return stacked_image
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_obs = process_default_observation(self, obs)
        self.obs_t1 = self.obs_t2
        self.obs_t2 = self.obs_t3
        self.obs_t3 = new_obs

        self.last_observation = self.get_observation()
        done = terminated or truncated
        return self.last_observation, reward, terminated, truncated,info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
    

if __name__ == "__main__":
    env = TetrisEnv()
    obs = env.reset()
    done = False
    action = env.action_space.sample()  # Replace with your policy or agent's action
    while not done:
        step_out = env.step(action)
        obs = step_out["observation"]
        #print(f"Action: {action}, {type(action)}")
        print(f"Observation: {obs.shape}, {type(obs)}")
        cv2.imshow("Tetris", obs)
        k = cv2.waitKey(0) & 0xFF
        # Check which key was pressed.
        if k == ord('w'):
            action = 1
        elif k == ord('d'):
            action = 2
        elif k == ord('a'):
            action = 3
        elif k == ord('s'):
            action = 4
        elif k == ord('q'):
            env.close()
            exit()
        else:
            action = 0
        done = step_out["done"]
    cv2.waitKey(0)
    env.close()
