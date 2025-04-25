from data_collector import get_collecter, get_replay_buffer
from networks import get_PPO_policy
from utils import select_device
from torchrl.objectives import PPOLoss, ClipPPOLoss
from Enviorments import get_tetris_env, get_mc_env
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import cv2
import os
import numpy as np
#https://pytorch.org/rl/main/tutorials/coding_ppo.html#policy

def train_ppo(get_env_func, env_name, lr=3e-4, frames_per_collector = 256, total_frames = 1_000_000, batches_to_store = 1024, mini_batch_size = 32, training_iter_per_batch = 10, gamma=0.99, lmbda=0.95, entropy_eps=1e-2, critic_coef=1, clip_grad=1.0):
    n_batches = math.ceil(total_frames / frames_per_collector)

    n_batches = math.ceil(n_batches / 10) * 10

    total_frames = n_batches * frames_per_collector

    save_dir = f"results/ppo_{env_name}_{lr=}_{total_frames=}_{mini_batch_size=}_{training_iter_per_batch=}_{entropy_eps=}/"
    os.makedirs(save_dir, exist_ok=True)
    device = select_device()

    ppo_policy, value_module = get_PPO_policy(get_env_func)
    ppo_policy.to(device)
    value_module.to(device)
    collector = get_collecter(get_env_func, ppo_policy, frames_per_collector, total_frames)
    replay_buffer = get_replay_buffer(batches_to_store, frames_per_collector, mini_batch_size)

    clip_epsilon = 0.2
    """loss_module = PPOLoss(
        actor_network=ppo_policy,
        critic_network=value_module,
        entropy_bonus=True,
        entropy_coef=entropy_eps,
        critic_coef=critic_coef,
        loss_critic_type="smooth_l1",
    ).to(device)"""
    loss_module = ClipPPOLoss(
        actor_network=ppo_policy,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=critic_coef,
        loss_critic_type="smooth_l1",
    )

   

    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_collector, 0.0
    )

    #batch_td = next(iter(collector))
    #print(f"Batch: {batch_td.shape}")
    #batch_td = batch_td.reshape(-1)
    #print(f"Batch: {batch_td.shape}")



    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    logs = defaultdict(list)
    out_file_txt = f"{save_dir}training.txt"
    with open(out_file_txt, "w") as f:
        f.write(save_dir + "\n")
    best_model = None
    best_model_score = -np.inf
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for collector_batch, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        objective_loss_acc = 0.0
        critic_loss_acc = 0.0
        entropy_loss_acc = 0.0
        for _ in range(training_iter_per_batch):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            for _ in range(frames_per_collector // mini_batch_size):
                mini_batch = replay_buffer.sample(mini_batch_size)
                #print(f"mini_batch: {mini_batch}")
                #input("Press enter to continue")
                loss_vals = loss_module(mini_batch.to(device))
                #print(f"loss_vals: {loss_vals}")
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                objective_loss_acc += loss_vals["loss_objective"].item()
                critic_loss_acc += loss_vals["loss_critic"].item()
                entropy_loss_acc += loss_vals["loss_entropy"].item()

                # Optimization: backward, grad clipping and optimization step
                optim.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(
                   loss_module.parameters(),
                   max_norm=clip_grad,
                )
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        logs["step_count"].append(tensordict_data["step_count"].float().mean().item())
        logs["lr"].append(optim.param_groups[0]["lr"])

        logs["loss objective"].append(objective_loss_acc)
        logs["loss critic"].append(critic_loss_acc)
        logs["loss entropy"].append(entropy_loss_acc)
        
        #print(f"{tensordict_data=}")
        out_str = f"Batch {collector_batch}/{total_frames//frames_per_collector}: reward:{logs['reward'][-1]}, step_count:{logs['step_count'][-1]}, lr:{logs['lr'][-1]}"
        #print(out_str, end='\n', flush=True)
        #with open(out_file_txt, "a") as f:
        #    f.write(out_str + "\n")
        if collector_batch % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                env = get_env_func()
                eval_rollout = env.rollout(1000, ppo_policy)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                total_reward =  eval_rollout["next", "reward"].sum().item()
                logs["eval reward (sum)"].append(
                   total_reward
                )
                if total_reward > best_model_score:
                    best_model_score = total_reward
                    best_model = ppo_policy.state_dict()
                    torch.save(best_model, f"{save_dir}best_model.pth")
                    print(f"Best model saved with score {best_model_score}")
                logs["eval step_count"].append(eval_rollout["step_count"].float().mean().item())
                del eval_rollout
            eval_str = f"Eval {collector_batch}/{total_frames//frames_per_collector}: avg eval reward:{logs['eval reward'][-1]}, sum eval reward :{logs['eval reward (sum)'][-1]}, AvgStepCount:{logs['eval step_count'][-1]}"
            print(eval_str, end='\n', flush=True)
            with open(out_file_txt, "a") as f:
                f.write(eval_str + "\n")
            

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()


    fig_dir = f"{save_dir}figures/"
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.savefig(f"{fig_dir}training_rewards.png")
    plt.close()             # closes the current figure


    plt.figure(figsize=(10, 10))
    plt.plot(logs["step_count"])
    plt.title("Avg step count (training)")
    plt.savefig(f"{fig_dir}AvgStepCount.png")
    plt.close()             # closes the current figure



    plt.figure(figsize=(10, 10))
    plt.plot(logs["eval reward (sum)"])
    plt.title("Eval Reward")
    plt.savefig(f"{fig_dir}SumEvalReward.png")
    plt.close()             # closes the current figure


    plt.figure(figsize=(10, 10))
    plt.plot(logs["eval step_count"])
    plt.title("Avg step count (test)")
    plt.savefig(f"{fig_dir}EvalStepCount.png")
    plt.close()             # closes the current figure



    plt.figure(figsize=(10, 10))
    plt.plot(logs["loss objective"])
    plt.title("Loss Objective")
    plt.savefig(f"{fig_dir}LossObjective.png")
    plt.close()             # closes the current figure

    plt.figure(figsize=(10, 10))
    plt.plot(logs["loss critic"])
    plt.title("Loss Critic")
    plt.savefig(f"{fig_dir}LossCritic.png")
    plt.close()             # closes the current figure

    plt.figure(figsize=(10, 10))
    plt.plot(logs["loss entropy"])
    plt.title("Loss Entropy")
    plt.savefig(f"{fig_dir}LossEntropy.png")
    plt.close()             # closes the current figure
    


    model_video_dir = f"{save_dir}model_video/"
    os.makedirs(model_video_dir, exist_ok=True)

    ppo_policy.load_state_dict(torch.load(f"{save_dir}best_model.pth", map_location=device))
    ppo_policy.eval()
    td = env.reset()
    for i in range(5):
        env = get_env_func()
        done = False
        video = []
        video_out = f"{model_video_dir}{i}.mp4"
        while not done:
            img = td["pixels"].squeeze(0).permute(1,2,0).cpu().numpy()
            video.append(img)
            cv2.imshow(f"Game {get_env_func}", img)
            cv2.waitKey(1)
            td = ppo_policy(td)
            td = env.step(td)['next']
            done = td["done"]
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")   # <─ THIS LINE
        w,h,c = video[0].shape
        writer = cv2.VideoWriter(video_out, fourcc, 30, (w, h), isColor=True)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {video_out}")
        for frame in video:
            frame = np.clip(frame, 0, 1) * 255 if frame.dtype.kind == "f" else frame
            frame = frame.astype(np.uint8, copy=False)
            writer.write(frame)
        writer.release()
        env.reset()
    return
        


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    train_ppo(get_mc_env, "MC", total_frames = 50_000)
    #train_ppo(get_tetris_env, "Tetris")