import torch
import cv2
import numpy as np
import random
import gym
from gym.spaces import Discrete, Box

# -----------------------------
# Global Settings and Constants
# -----------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30  # Pixel size for the human (display) view

# -----------------------------
# Tetromino Definitions (as boolean matrices)
# -----------------------------
tetrominoes = {
    1: torch.tensor([[True, True, True, True]], dtype=torch.bool),  # I piece (1x4)
    2: torch.tensor([[True, False],
                     [True, True],
                     [False, True]], dtype=torch.bool),                # J piece (3x2)
    3: torch.tensor([[False, True],
                     [True, True],
                     [True, False]], dtype=torch.bool),                # L piece (3x2)
    4: torch.tensor([[True, True],
                     [True, True]], dtype=torch.bool),                 # O piece (2x2)
    5: torch.tensor([[False, True, True],
                     [True, True, False]], dtype=torch.bool),          # S piece (2x3)
    6: torch.tensor([[True, True, True],
                     [False, True, False]], dtype=torch.bool),         # T piece (2x3)
    7: torch.tensor([[True, True, False],
                     [False, True, True]], dtype=torch.bool)           # Z piece (2x3)
}

def new_piece():
    """Creates a new tetromino piece with a random type and starting position."""
    type_val = random.randint(1, 7)
    shape = tetrominoes[type_val].clone()  # Boolean matrix for the piece
    h, w = shape.shape
    # Position the piece so that its top-left lies above the board, centered horizontally.
    pos = torch.tensor([-h, (BOARD_WIDTH - w) // 2], dtype=torch.int64)
    return {'type': type_val, 'shape': shape, 'pos': pos}

def rotate_piece(piece):
    """
    Rotates the given tetromino piece 90° clockwise.
    (Uses torch.rot90 with k=3, which is equivalent.)
    """
    return torch.rot90(piece['shape'], k=3, dims=(0, 1))

def check_collision(board, piece):
    """
    Checks whether the piece (with its boolean shape and current position)
    collides with board boundaries or any frozen (True) cells.
    """
    pos = piece['pos']
    shape = piece['shape']
    h, w = shape.shape
    for i in range(h):
        for j in range(w):
            if shape[i, j]:
                r = pos[0] + i
                c = pos[1] + j
                if c < 0 or c >= BOARD_WIDTH:
                    return True
                if r >= BOARD_HEIGHT:
                    return True
                if r >= 0 and board[r, c]:
                    return True
    return False

def freeze_piece(board, piece):
    """
    Freezes the current piece onto the board.
    Sets board cell to True wherever the piece's boolean shape is True.
    """
    pos = piece['pos']
    shape = piece['shape']
    h, w = shape.shape
    for i in range(h):
        for j in range(w):
            if shape[i, j]:
                r = pos[0] + i
                c = pos[1] + j
                if r >= 0:
                    board[r, c] = True
    return board

def clear_lines(board):
    """
    Clears full lines from the board.
    Returns the new board and the number of cleared lines.
    """
    new_rows = []
    lines_cleared = 0
    for r in range(BOARD_HEIGHT):
        if board[r, :].all().item():
            lines_cleared += 1
        else:
            new_rows.append(board[r, :])
    for _ in range(lines_cleared):
        empty_row = torch.zeros(BOARD_WIDTH, dtype=torch.bool)
        new_rows.insert(0, empty_row)
    new_board = torch.stack(new_rows)
    return new_board, lines_cleared

# -----------------------------
# Gym-style Tetris Environment
# -----------------------------
class TetrisEnv:
    def __init__(self):
        self.board_width = BOARD_WIDTH
        self.board_height = BOARD_HEIGHT
        self.block_size = BLOCK_SIZE
        self.reset()  # Initialize game state
        
        # Define the gym spaces.
        # There are 4 actions: 0: move left, 1: move right, 2: rotate, 3: soft drop.
        self.action_space = Discrete(4)
        # Observation: float values in [0,1] in shape (BOARD_HEIGHT, BOARD_WIDTH, 3).
        self.observation_space = Box(low=0.0, high=1.0, 
                                     shape=(self.board_height, self.board_width, 3), 
                                     dtype=np.float32)
    
    def reset(self):
        """Resets the game state and returns the initial observation as a float tensor."""
        # self.board holds frozen pieces as a boolean tensor.
        self.board = torch.zeros((self.board_height, self.board_width), dtype=torch.bool)
        self.current_piece = new_piece()
        self.game_over = False
        self.total_lines_cleared = 0
        return self._get_obs_tensor()
    
    def step(self, action):
        """
        Applies the given action and then advances the game state by one gravity drop.
        
        Actions:
            0: Move left.
            1: Move right.
            2: Rotate piece.
            3: Soft drop.
        
        Returns:
            observation: A torch float tensor of shape (1, 3, BOARD_HEIGHT, BOARD_WIDTH) (values in [0,1])
            reward: A torch float tensor scalar (lines cleared)
            done: A torch boolean tensor scalar (game over flag)
            info: A dict with tensor values (empty here)
        """
        reward = 0

        # --- Apply Agent Action ---
        if action == 0:  # Move left.
            self.current_piece['pos'][1] -= 1
            if check_collision(self.board, self.current_piece):
                self.current_piece['pos'][1] += 1
        elif action == 1:  # Move right.
            self.current_piece['pos'][1] += 1
            if check_collision(self.board, self.current_piece):
                self.current_piece['pos'][1] -= 1
        elif action == 2:  # Rotate.
            old_shape = self.current_piece['shape'].clone()
            self.current_piece['shape'] = rotate_piece(self.current_piece)
            if check_collision(self.board, self.current_piece):
                self.current_piece['shape'] = old_shape
        elif action == 3:  # Soft drop.
            self.current_piece['pos'][0] += 1
            if check_collision(self.board, self.current_piece):
                self.current_piece['pos'][0] -= 1
        
        # --- Gravity Drop ---
        self.current_piece['pos'][0] += 1
        if check_collision(self.board, self.current_piece):
            self.current_piece['pos'][0] -= 1
            self.board = freeze_piece(self.board, self.current_piece)
            self.board, lines_cleared = clear_lines(self.board)
            reward += lines_cleared
            self.total_lines_cleared += lines_cleared
            self.current_piece = new_piece()
            if check_collision(self.board, self.current_piece):
                self.game_over = True
        
        obs_tensor = self._get_obs_tensor()  # Shape: (1, 3, H, W)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(self.game_over, dtype=torch.bool)
        info_tensor = {}  # Additional info (empty for now)
        return obs_tensor, reward_tensor, done_tensor, info_tensor

    def _get_obs_tensor(self):
        """
        Constructs and returns the observation as a torch float tensor of shape (1, 3, BOARD_HEIGHT, BOARD_WIDTH).
        Colors (normalized to [0,1]):
          - Frozen (placed) cells: white → (1, 1, 1)
          - Empty cells: black → (0, 0, 0)
          - Moving piece: red → (1, 0, 0)
        """
        # Build an RGB image (H, W, 3) using uint8 values first.
        input_img = np.zeros((self.board_height, self.board_width, 3), dtype=np.uint8)
        board_np = self.board.numpy()
        # Set frozen cells to white.
        for r in range(self.board_height):
            for c in range(self.board_width):
                if board_np[r, c]:
                    input_img[r, c, :] = [255, 0, 0]
                else:
                    input_img[r, c, :] = [0, 255, 0]
        # Overlay the moving piece in red.
        pos = self.current_piece['pos']
        shape = self.current_piece['shape'].numpy()
        h_shape, w_shape = shape.shape
        for i in range(h_shape):
            for j in range(w_shape):
                if shape[i, j]:
                    r = pos[0] + i
                    c = pos[1] + j
                    if 0 <= r < self.board_height and 0 <= c < self.board_width:
                        input_img[r, c, :] = [0, 0, 255]
        # Convert the image to float32 and normalize to [0,1]
        input_img = input_img.astype(np.float32) / 255.0
        # Convert from H x W x 3 to tensor shape (1, 3, H, W)
        obs_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0)
        return obs_tensor

    def render(self, mode='human'):
        """
        Renders the current game state in two views:
        
          1. "human": A scaled-up view (using BLOCK_SIZE per cell) for display via OpenCV.
                     Frozen cells are shown in white; the moving piece is red.
          2. "input": The raw observation image (BOARD_HEIGHT x BOARD_WIDTH x 3, float in [0,1])
                     converted to 8-bit for display.
        
        Returns:
            A dictionary with keys "human" and "input" containing the corresponding images.
        """
        # --- Human View (scaled) ---
        human_img = np.zeros((self.board_height * self.block_size, 
                              self.board_width * self.block_size, 3), dtype=np.uint8)
        # Draw frozen cells as white.
        for r in range(self.board_height):
            for c in range(self.board_width):
                if self.board[r, c]:
                    cv2.rectangle(human_img,
                                  (c * self.block_size, r * self.block_size),
                                  ((c + 1) * self.block_size, (r + 1) * self.block_size),
                                  (255, 255, 255), -1)
        # Draw moving piece as red.
        pos = self.current_piece['pos']
        shape = self.current_piece['shape']
        h_shape, w_shape = shape.shape
        for i in range(h_shape):
            for j in range(w_shape):
                if shape[i, j]:
                    r = pos[0] + i
                    c = pos[1] + j
                    if 0 <= r < self.board_height and 0 <= c < self.board_width:
                        cv2.rectangle(human_img,
                                      (int(c * self.block_size), int(r * self.block_size)),
                                      (int((c + 1) * self.block_size), int((r + 1) * self.block_size)),
                                      (0, 0, 255), -1)
        # Draw grid lines.
        for r in range(self.board_height + 1):
            cv2.line(human_img, (0, r * self.block_size), 
                     (self.board_width * self.block_size, r * self.block_size), (0, 0, 0), 1)
        for c in range(self.board_width + 1):
            cv2.line(human_img, (c * self.block_size, 0), 
                     (c * self.block_size, self.board_height * self.block_size), (0, 0, 0), 1)
        
        # --- Input View (raw observation) ---
        # Get the observation tensor, convert to (H, W, 3) float and then to 0-255 uint8 for display.
        input_tensor = self._get_obs_tensor().squeeze(0).permute(1, 2, 0)  # shape (H, W, 3)
        input_img = (input_tensor.numpy() * 255).astype(np.uint8)
        
        if mode == 'human':
            cv2.imshow("Tetris - Human", human_img)
            cv2.imshow("Tetris - Input", input_img)
            cv2.waitKey(1)
        
        return {"human": human_img, "input": input_img}

    def close(self):
        """Closes any OpenCV windows."""
        cv2.destroyAllWindows()

# -----------------------------
# Main - Human Controlled Tetris via WASD
# -----------------------------
if __name__ == '__main__':
    env = TetrisEnv()
    obs = env.reset()
    done = False

    print("Control Tetris using keys:")
    print("  A: Move left")
    print("  D: Move right")
    print("  W: Rotate piece")
    print("  S: Soft drop")
    print("  Q: Quit game")

    while not done:
        env.render(mode='human')
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            action = 0
        elif key == ord('d'):
            action = 1
        elif key == ord('w'):
            action = 2
        elif key == ord('s'):
            action = 3
        else:
            continue
        
        obs, reward, done, info = env.step(action)

    print("Game over! Total lines cleared:", env.total_lines_cleared)
    env.close()
