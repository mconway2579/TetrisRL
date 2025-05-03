from data_collector import get_collecter, get_replay_buffer
from networks import get_PPO_policy
from utils import select_device, record_video, graph_logs
from torchrl.objectives import PPOLoss, ClipPPOLoss
from Enviorments import get_tetris_env, get_mcd_env, get_tetris_env_flat
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import cv2
import os
import numpy as np
import time
#https://pytorch.org/rl/main/tutorials/coding_ppo.html#policy


def train_ppo(get_env_func, env_name, lr=1e-5, frames_per_collector=256, total_frames=1_000_000, batches_to_store=2048, mini_batch_size=128, training_iter_per_batch=10, gamma=0.99, lmbda=0.95, entropy_eps=1e-4, critic_coef=1, clip_grad=1):
    n_batches = np.ceil(total_frames / frames_per_collector)

    n_batches = np.ceil(n_batches / 10) * 10

    total_frames = (n_batches * frames_per_collector) +1

    save_dir = f"results/ppo_{env_name}_{lr=}_{total_frames=}_{mini_batch_size=}_{training_iter_per_batch=}_{entropy_eps=}/"
    os.makedirs(save_dir, exist_ok=True)
    device = select_device()

    ppo_policy, value_module = get_PPO_policy(get_env_func)
    ppo_policy.to(device)
    value_module.to(device)
    collector = get_collecter(get_env_func, ppo_policy, frames_per_collector, total_frames)
    replay_buffer = get_replay_buffer(batches_to_store, frames_per_collector, mini_batch_size)

    clip_epsilon = 0.2
    
    loss_module = ClipPPOLoss(
        actor_network=ppo_policy,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=critic_coef,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
        value_clip=True
    )
   

    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)



    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    logs = defaultdict(list)
    out_file_txt = f"{save_dir}training.txt"
    with open(out_file_txt, "w") as f:
        f.write(save_dir + "\n")
        f.write(f"{ppo_policy}\n")
    best_model_reward_sum = -np.inf
    best_model_reward_avg = -np.inf
    best_model_step_count = -np.inf
    start_time = time.time()
    last_eval_time = time.time()
    for collector_batch, tensordict_data in enumerate(collector):
        if len(tensordict_data["action"].shape) == 2 and tensordict_data["action"].shape[1] == 1:
            tensordict_data["action"] = tensordict_data["action"].squeeze(1)
        objective_loss_acc = 0.0
        critic_loss_acc = 0.0
        entropy_loss_acc = 0.0
        for _ in range(training_iter_per_batch):

            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            idx = replay_buffer.extend(data_view)
            #print(f"{data_view=}")
            rewards     = data_view["next", "reward"].squeeze(-1)        # [B]
            dones       = data_view["next", "done"].squeeze(-1).float()  # [B]
            values      = data_view["state_value"].squeeze(-1)           # V(s_t)  [B]
            next_values = data_view["next", "state_value"].squeeze(-1)   # V(s_{t+1})  [B]
            td_target = rewards + gamma * next_values * (1.0 - dones)
            td_error = td_target - values   
            replay_buffer.update_priority(idx, td_error.abs())
            for _ in range(frames_per_collector // mini_batch_size):
                mini_batch, info = replay_buffer.sample(mini_batch_size, return_info=True)
                mini_batch = mini_batch.to(device)
                advantage_module(mini_batch)

                loss_vals = loss_module(mini_batch)
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

                optim.step()

                rewards     = mini_batch["next", "reward"].squeeze(-1)        # [B]
                dones       = mini_batch["next", "done"].squeeze(-1).float()  # [B]
                values      = mini_batch["state_value"].squeeze(-1)           # V(s_t)  [B]
                next_values = mini_batch["next", "state_value"].squeeze(-1)   # V(s_{t+1})  [B]
                td_target = rewards + gamma * next_values * (1.0 - dones)
                td_error = td_target - values
                #print(f"TD Error: {td_error.max()}, {td_error.min()}, {td_error.mean()}")

                replay_buffer.update_priority(info["index"], td_error.abs())
                # optim.zero_grad()

        logs["collector avg_reward"].append(tensordict_data["next", "reward"].mean().item())
        max_step_count = tensordict_data["step_count"].max().item()
        #print(f"MaxStepCount: {max_step_count}")
        logs["collector max step count"].append(max_step_count)
        logs["lr"].append(optim.param_groups[0]["lr"])

        logs["loss objective"].append(objective_loss_acc)
        logs["loss critic"].append(critic_loss_acc)
        logs["loss entropy"].append(entropy_loss_acc)
        
     
        if collector_batch % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                env = get_env_func()
                rollouts = []
                for i in range(5):
                    env.reset()
                    rollout = env.rollout(1000, ppo_policy)
                    rollouts.append(rollout)
                env.reset()
                eval_rollout = torch.cat(rollouts, dim=0)

                sum_reward = eval_rollout["next", "reward"].sum().item()
                avg_reward = eval_rollout["next", "reward"].mean().item()
                total_steps = eval_rollout["step_count"].sum().item()

                logs["eval avg reward"].append(avg_reward)
                logs["eval sum reward"].append(sum_reward)
                logs["eval step_count"].append(total_steps)
                if sum_reward >= best_model_reward_sum:
                    best_model_reward_sum = sum_reward
                    torch.save(ppo_policy.state_dict(), f"{save_dir}best_model_reward_sum.pth")
                    torch.save(value_module.state_dict(), f"{save_dir}best_model_reward_sum_value.pth")
                    with open(out_file_txt, "a") as f:
                        f.write(f"Best model saved with reward sum {best_model_reward_sum}\n")
                    print(f"Best model saved with reward sum {best_model_reward_sum}")

                if avg_reward >= best_model_reward_avg:
                    best_model_reward_avg = avg_reward
                    torch.save(ppo_policy.state_dict(), f"{save_dir}best_model_reward_avg.pth")
                    torch.save(value_module.state_dict(), f"{save_dir}best_model_reward_avg_value.pth")
                    with open(out_file_txt, "a") as f:
                        f.write(f"Best model saved with reward avg {best_model_reward_avg}\n")
                    print(f"Best model saved with reward avg {best_model_reward_avg}")

                if total_steps >= best_model_step_count:
                    best_model_step_count = total_steps
                    torch.save(ppo_policy.state_dict(), f"{save_dir}best_step_count_model.pth")
                    torch.save(value_module.state_dict(), f"{save_dir}best_step_count_model_value.pth")
                    with open(out_file_txt, "a") as f:
                        f.write(f"Best model saved with step count {best_model_step_count}\n")
                        print(f"Best model saved with step count {best_model_step_count}")
                logs["eval step_count"].append(total_steps)
                del eval_rollout
            eval_str = f"Eval {collector_batch//10}/{(total_frames//frames_per_collector)//10}: avg eval reward:{logs['eval avg reward'][-1]}, sum eval reward :{logs['eval sum reward'][-1]}, StepCount:{logs['eval step_count'][-1]}, total_time:{time.time()-start_time:.2f} seconds, Time since last eval: {time.time()-last_eval_time:.2f} seconds"
            last_eval_time = time.time()
            print(eval_str, end='\n', flush=True)
            with open(out_file_txt, "a") as f:
                f.write(eval_str + "\n")
            

    fig_dir = f"{save_dir}figures/"
    os.makedirs(fig_dir, exist_ok=True)
    graph_logs(logs, fig_dir)


    initial_state_dict = {k: v.detach().cpu().clone().to(device) for k, v in ppo_policy.state_dict().items()}


    model_video_dir = f"{save_dir}sum_rewards_model_video/"
    os.makedirs(model_video_dir, exist_ok=True)


    ppo_policy.load_state_dict(torch.load(f"{save_dir}best_model_reward_sum.pth", map_location=device))
    assert any(
        not torch.allclose(new, initial_state_dict[k], atol=1e-6)
        for k, new in ppo_policy.state_dict().items()
    ), "Policy parameters did not update!"
    ppo_policy.eval()
    td = env.reset()
    for i in range(5):
        env = get_env_func()
        record_video(env, ppo_policy, f"{model_video_dir}{i}.mp4")

    
    model_video_dir = f"{save_dir}avg_rewards_model_video/"
    os.makedirs(model_video_dir, exist_ok=True)
    ppo_policy.load_state_dict(torch.load(f"{save_dir}best_model_reward_avg.pth", map_location=device))
    assert any(
        not torch.allclose(new, initial_state_dict[k], atol=1e-6)
        for k, new in ppo_policy.state_dict().items()
    ), "Policy parameters did not update!"

    ppo_policy.eval()
    td = env.reset()
    for i in range(5):
        env = get_env_func()
        record_video(env, ppo_policy, f"{model_video_dir}{i}.mp4")


    model_video_dir = f"{save_dir}best_step_count_model_video/"
    os.makedirs(model_video_dir, exist_ok=True)

    ppo_policy.load_state_dict(torch.load(f"{save_dir}best_step_count_model.pth", map_location=device))
    assert any(
        not torch.allclose(new, initial_state_dict[k], atol=1e-6)
        for k, new in ppo_policy.state_dict().items()
    ), "Policy parameters did not update!"

    ppo_policy.eval()
    td = env.reset()
    for i in range(5):
        env = get_env_func()
        record_video(env, ppo_policy, f"{model_video_dir}{i}.mp4")
    return
        


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    #train_ppo(get_mcc_env, "MCC", total_frames = 50_000, frames_per_collector = 128)
    # train_ppo(get_mcd_env, "MCd", total_frames = 10_000)

    #train_ppo(get_tetris_env, "tetris", total_frames = 10_000)
    train_ppo(get_tetris_env_flat, "tetris_flat", total_frames = 50_000)
