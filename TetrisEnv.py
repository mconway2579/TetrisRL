import numpy as np
import cv2
import random
import gym
from gym.spaces import Discrete, Box
import torch
from tensordict import TensorDict
from torchrl.envs import GymWrapper

# -----------------------------
# Global Settings and Constants
# -----------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30  # Pixel size for the human (display) view

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
    t = random.randint(1, 7) #4
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
            shape=(1, self.board_h, self.board_w, 3),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_h, self.board_w), dtype=bool)
        self.current_piece = new_piece()
        self.game_over = False
        self.total_lines = 0
        obs = self._get_obs()            # shape (H, W, 3)
        return obs[None, ...], {}        # shape (1, H, W, 3)

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

        obs = self._get_obs()            # shape (H, W, 3)
        return obs[None, ...], reward, self.game_over, False, {}

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
        return img.astype(np.float32) / 255.0

    def render(self):
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
        
        cv2.imshow("Tetris - Human", human)
        cv2.imshow("Tetris - Input", inp)
        cv2.waitKey(1)
        return {"human": human, "input": inp}

    def close(self):
        cv2.destroyAllWindows()

def get_env():
    # 1) build wrapped env with integer actions
    base_env = TetrisEnv()
    env = GymWrapper(
        base_env,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        categorical_action_encoding=True,
        from_pixels=False,
    )
    return env

if __name__ == '__main__':
    env = get_env()

    # 2) reset → TensorDict
    td = env.reset()
    print(f"reset: {td=}")
    obs = td["observation"]              # Tensor(shape=[1,20,10,3])
    done = td["done"].item()
    total_lines = 0

    print(f"{obs.shape=}")               # (1,20,10,3)
    print("Controls: A=left, D=right, W=rotate, S=soft-drop, Q=quit")

    # 3) loop: render + read key + step via tensordict
    while not done:
        env.render()

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        mapping = {ord('a'): 0, ord('d'): 1, ord('w'): 2, ord('s'): 3}
        if key not in mapping:
            continue
        action_int = mapping[key]

        in_td = TensorDict(
            {"action": torch.tensor(action_int, dtype=torch.int64, device=env.device)},
            batch_size=[]
        )
        print(f"action: {in_td=}")

        out_td = env.step(in_td)
        print(f"step: {out_td=}")
        next_td = out_td["next"]
        print(f"next: {next_td=}")

        obs    = next_td["observation"]
        reward = next_td["reward"].item()
        done   = next_td["done"].item()
        total_lines += reward

    print("Game over! Lines cleared:", total_lines)
    env.close()
