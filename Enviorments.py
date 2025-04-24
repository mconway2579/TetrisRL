import numpy as np
import cv2
import random
import gym
from gym.spaces import Discrete, Box
import torch
from tensordict import TensorDict
from torchrl.envs import GymWrapper
from torchrl.envs.utils import check_env_specs
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
import utils
# -----------------------------
# Global Settings and Constants
# -----------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30  # Pixel size for the human (display) view
device = utils.select_device()
# -----------------------------
# Tetromino Definitions (as NumPy boolean arrays)
# -----------------------------
tetrominoes = {
    1: np.array([[ True,  True,  True,  True]], dtype=bool),   # I piece (1×4)
    2: np.array([[ True, False],
                 [ True,  True],
                 [False,  True]], dtype=bool),                 # J piece (3×2)
    3: np.array([[False,  True],
                 [ True,  True],
                 [ True, False]], dtype=bool),                 # L piece (3×2)
    4: np.array([[ True,  True],
                 [ True,  True]], dtype=bool),                 # O piece (2×2)
    5: np.array([[False,  True,  True],
                 [ True,  True, False]], dtype=bool),          # S piece (2×3)
    6: np.array([[ True,  True,  True],
                 [False,  True, False]], dtype=bool),          # T piece (2×3)
    7: np.array([[ True,  True, False],
                 [False,  True,  True]], dtype=bool),          # Z piece (2×3)
}

def new_piece():
    """Create a new random tetromino with its shape and spawn position."""
    t = 4#random.randint(1, 7) #4
    shape = tetrominoes[t].copy()
    h, w = shape.shape
    pos = np.array([-h, (BOARD_WIDTH - w) // 2], dtype=int)
    return {'type': t, 'shape': shape, 'pos': pos}

def rotate_piece(piece):
    """Rotate shape 90° clockwise."""
    return np.rot90(piece['shape'], k=-1)

def check_collision(board, piece):
    """Return True if piece overlaps walls, floor, or frozen blocks."""
    pos = piece['pos']
    shape = piece['shape']
    h, w = shape.shape
    for i in range(h):
        for j in range(w):
            if not shape[i, j]:
                continue
            r, c = pos[0] + i, pos[1] + j
            if c < 0 or c >= BOARD_WIDTH or r >= BOARD_HEIGHT:
                return True
            if r >= 0 and board[r, c]:
                return True
    return False

def freeze_piece(board, piece):
    """Stamp the piece into the board as frozen cells."""
    pos = piece['pos']
    shape = piece['shape']
    h, w = shape.shape
    for i in range(h):
        for j in range(w):
            if shape[i, j]:
                r, c = pos[0] + i, pos[1] + j
                if r >= 0:
                    board[r, c] = True
    return board

def clear_lines(board):
    """Remove full rows, shift everything down, return new board + count."""
    new_rows = []
    cleared = 0
    for r in range(BOARD_HEIGHT):
        if board[r].all():
            cleared += 1
        else:
            new_rows.append(board[r].copy())
    if cleared:
        empty = np.zeros((cleared, BOARD_WIDTH), dtype=bool)
        board = np.vstack([empty] + new_rows)
    return board, cleared

class TetrisEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_w = BOARD_WIDTH
        self.board_h = BOARD_HEIGHT
        self.block_size = BLOCK_SIZE

        self.action_space = Discrete(4)
        # batch_size=1, H×W×3 float32 image in [0,1]
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(3, self.board_h, self.board_w),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_h, self.board_w), dtype=bool)
        self.current_piece = new_piece()
        self.game_over = False
        self.total_lines = 0
        obs = self._get_obs()            # shape (H, W, 3)
        return obs[...], {}        # shape (1, H, W, 3)

    def step(self, action):
        reward = 0
        # Agent action
        if action == 0:  # left
            self.current_piece['pos'][1] -= 1
            if check_collision(self.board, self.current_piece):
                self.current_piece['pos'][1] += 1
        elif action == 1:  # right
            self.current_piece['pos'][1] += 1
            if check_collision(self.board, self.current_piece):
                self.current_piece['pos'][1] -= 1
        elif action == 2:  # rotate
            old = self.current_piece['shape'].copy()
            self.current_piece['shape'] = rotate_piece(self.current_piece)
            if check_collision(self.board, self.current_piece):
                self.current_piece['shape'] = old
        elif action == 3:  # soft drop
            self.current_piece['pos'][0] += 1
            if check_collision(self.board, self.current_piece):
                self.current_piece['pos'][0] -= 1

        # Gravity
        self.current_piece['pos'][0] += 1
        if check_collision(self.board, self.current_piece):
            self.current_piece['pos'][0] -= 1
            self.board = freeze_piece(self.board, self.current_piece)
            self.board, lines = clear_lines(self.board)
            reward += lines
            self.total_lines += lines
            self.current_piece = new_piece()
            if check_collision(self.board, self.current_piece):
                self.game_over = True
        if self.board[0].any():    # any cell in top row is occupied
            self.game_over = True
        
        # Negative reward for holes (empty cells with filled cells above them)
        n_holes = 0
        for col in range(self.board_w):
            filled_found = False
            for row in range(self.board_h):
                if self.board[row, col]:
                    filled_found = True
                elif filled_found:
                    n_holes += 1
        # Find the height of the highest pixel (lowest y value)
        highest_pixel = self.board_h  # Initialize to max height
        for row in range(self.board_h):
            if self.board[row, :].any():
                highest_pixel = min(highest_pixel, row)
        height = self.board_h - highest_pixel
        reward += (-0.36 * n_holes) + (-0.51*height) + (-1*self.game_over)

        obs = self._get_obs()            # shape (H, W, 3)
        return obs[...], reward, self.game_over, False, {}

    def _get_obs(self):
        img = np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8)
        # frozen cells = white
        for r in range(self.board_h):
            for c in range(self.board_w):
                if self.board[r, c]:
                    img[r, c] = [255, 0, 0]
                else:
                    img[r, c] = [0, 255, 0]
        # moving piece = red
        pos, shape = self.current_piece['pos'], self.current_piece['shape']
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    rr, cc = pos[0] + i, pos[1] + j
                    if 0 <= rr < self.board_h and 0 <= cc < self.board_w:
                        img[rr, cc] = [0, 0, 255]
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32) / 255.0

    def render(self, mode='human'):
        human = np.zeros((self.board_h*self.block_size,
                          self.board_w*self.block_size, 3), dtype=np.uint8)
        for r in range(self.board_h):
            for c in range(self.board_w):
                if self.board[r, c]:
                    cv2.rectangle(
                        human,
                        (c*self.block_size, r*self.block_size),
                        ((c+1)*self.block_size, (r+1)*self.block_size),
                        (255, 255, 255), -1
                    )
        pos, shape = self.current_piece['pos'], self.current_piece['shape']
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    rr, cc = pos[0]+i, pos[1]+j
                    if 0 <= rr < self.board_h and 0 <= cc < self.board_w:
                        cv2.rectangle(
                            human,
                            (cc*self.block_size, rr*self.block_size),
                            ((cc+1)*self.block_size, (rr+1)*self.block_size),
                            (0, 0, 255), -1
                        )
        # grid
        for i in range(self.board_h+1):
            cv2.line(human, (0, i*self.block_size),
                     (self.board_w*self.block_size, i*self.block_size), (0,0,0), 1)
        for j in range(self.board_w+1):
            cv2.line(human, (j*self.block_size, 0),
                     (j*self.block_size, self.board_h*self.block_size), (0,0,0), 1)

        inp = (self._get_obs() * 255).astype(np.uint8)
        inp = np.transpose(inp, (1, 2, 0))
        #cv2.imshow("Tetris - Human", human)
        #cv2.imshow("Tetris - Input", inp)
        #cv2.waitKey(1)
        return inp

    def close(self):
        cv2.destroyAllWindows()

def get_tetris_env():
    # 1) build wrapped env with integer actions
    base_env = TetrisEnv()
    wrapped_env = GymWrapper(
        base_env,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        categorical_action_encoding=True,
        from_pixels=False,
        auto_reset=False
    )
    transformed_env = TransformedEnv(
        wrapped_env,
        StepCounter(step_count_key="step_count"),   # <- adds the key every step :contentReference[oaicite:0]{index=0}
        DoubleToFloat(),                            # typical preprocessing
    ).to(device)
    return transformed_env


from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    ToTensorImage,
    GrayScale,
    Resize,
    RenameTransform
)
def get_pong_env():
    base_env = GymEnv("ALE/Pong-v5", device=device, render_mode="rgb_array")
    env = TransformedEnv(
        base_env,
        Compose(
            ToTensorImage(),   # uint8 H×W×C → float32 C×H×W
            Resize((84, 84)),  # downsample to 84×84
            StepCounter(),
            RenameTransform(
                in_keys=["pixels"],                # original key
                out_keys=["observation"],          # new key
                create_copy=False                  # move rather than copy
            ),
            
        )
    )
    return env


def get_mc_env():
    base_env = GymEnv("MountainCarContinuous-v0", device=device, render_mode="rgb_array")
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter()
        )
    )
    return env

if __name__ == '__main__':
    print(list(GymEnv.available_envs))
    env_name = "MC"
    env = None
    action_mapping = None
    if env_name == "Tetris":
        env = get_tetris_env()
        action_mapping = {ord('a'): 0, ord('d'): 1, ord('w'): 2, ord('s'): 3}
    elif env_name == "Pong":
        env = get_pong_env()
        action_mapping = {ord('a'): 0, ord('d'): 1, ord('w'): 2, ord('s'): 3}
    elif env_name == "MC":
        env = get_mc_env()
        action_mapping = {ord('a'): [-1.0], ord('d'):[ 1.0]}
    else:
        raise ValueError("Unknown environment name")
    check_env_specs(env)
    print(f"{env.action_space=}")
    rollout = env.rollout(3)

    # 2) reset → TensorDict
    td = env.reset()
    print(f"reset: {td=}")
    obs = td["observation"]
    done = td["done"].item()
    total_lines = 0

    print(f"{obs.shape=}")               # (1,20,10,3)
    print("Controls: A=left, D=right, W=rotate, S=soft-drop, Q=quit")

    # 3) loop: render + read key + step via tensordict
    while True:
        pixels = env.render()
        print(f"{pixels.shape=}")
        cv2.imshow("Window", pixels)
        cv2.waitKey(1)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key not in action_mapping:
            continue
        action_rep = action_mapping[key]
        td["action"] = action_rep

        print(f"action: {td=}")

        out_td = env.step(td)
        print(f"step: {out_td=}")
        next_td = out_td["next"]
        print(f"next: {next_td=}")

        obs    = next_td["observation"]
        reward = next_td["reward"].item()
        done   = next_td["done"].item()
        total_lines += reward
        print(f"{done = }")
        if done:
            env.reset()

    print("Game over! Lines cleared:", total_lines)
    env.close()
