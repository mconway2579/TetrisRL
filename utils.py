import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")          # purely file-based; no window

import os
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def record_video(env, policy,  video_path):
    """Record a video of the environment."""
    td = env.reset()
    done = False
    video = []
    max_frames = 5000
    while not done and len(video) < max_frames:
        img = td["pixels"].squeeze(0).permute(1,2,0).cpu().numpy()
        video.append(img)
        cv2.imshow(f"Game {video_path}", img)
        cv2.waitKey(1)
        td = policy(td)
        td = env.step(td)['next']
        done = td["done"]
    cv2.destroyAllWindows()
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")   # <─ THIS LINE
    w,h,c = video[0].shape
    writer = cv2.VideoWriter(video_path, fourcc, 30, (h, w), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")
    for frame in video:
        frame = np.clip(frame, 0, 1) * 255 if frame.dtype.kind == "f" else frame
        frame = frame.astype(np.uint8, copy=False)
        writer.write(frame)
    writer.release()
    env.reset()

def graph_logs(logs, save_dir):
    """Graph the logs."""
    os.makedirs(save_dir, exist_ok=True)

    for k, v in logs.items():
        print(f"{k=}, {v[0]=}")
        plt.figure(figsize=(10, 10))
        plt.plot(v)
        plt.title(f"{k}")
        plt.savefig(f"{save_dir}{k}.png")
        plt.close()             # closes the current figure