from data_collector import get_collecter, get_replay_buffer
from utils import select_device, record_video, graph_logs
from Enviorments import get_tetris_env, get_mcd_env, get_tetris_env_flat
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import cv2
import os
import numpy as np
import time
# for dqn
from torchrl.objectives import DQNLoss, SoftUpdate

from networks import get_EGDQN  # Use get_EGDQN from the second code snippet

#https://pytorch.org/rl/main/tutorials/coding_ppo.html#policy


def train_dqn(get_env_func, env_name, lr=1e-5, frames_per_collector=256, total_frames=1_000_000, batches_to_store=2048, mini_batch_size=128, training_iter_per_batch=10, gamma=0.999,  clip_grad=1, eps_start=1.0, eps_end=0.05):
    n_batches = np.ceil(total_frames / frames_per_collector)
    n_batches = np.ceil(n_batches / 10) * 10
    total_frames = (n_batches * frames_per_collector) +1
    save_dir = f"results/dqn_{env_name}_{lr=}_{total_frames=}_{mini_batch_size=}_{training_iter_per_batch=}_{eps_start=}_{eps_end=}/"
    os.makedirs(save_dir, exist_ok=True)
    device = select_device()
    print(f"\n\n\n")
    print(f"Begin {save_dir}")
    actor, actor_explore = get_EGDQN(get_env_func, eps_start, eps_end, total_frames)
    actor.to(device)
    actor_explore.to(device)
    collector = get_collecter(get_env_func, actor, frames_per_collector, total_frames)

    replay_buffer = get_replay_buffer(batches_to_store, frames_per_collector, mini_batch_size)
    
    # Define DQN loss
    env = get_env_func()
    loss_module = DQNLoss(actor, action_space=env.action_spec, double_dqn = True, delay_value=True, loss_function="smooth_l1").to(device)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, tau=0.5) #soft update of the target network
    

   

    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

    logs = defaultdict(list)
    out_file_txt = f"{save_dir}training.txt"
    with open(out_file_txt, "w") as f:
        f.write(save_dir + "\n")
        f.write(f"{actor=}\n")
    best_model_reward_sum = -np.inf
    best_model_reward_avg = -np.inf
    best_model_step_count = -np.inf
    best_model_lines_cleared = -np.inf
    start_time = time.time()
    last_eval_time = time.time()
    for collector_batch, tensordict_data in enumerate(collector):
        if len(tensordict_data["action"].shape) == 2 and tensordict_data["action"].shape[1] == 1:
            tensordict_data["action"] = tensordict_data["action"].squeeze(1)
        dqn_loss_acc = 0.0
        data_view = tensordict_data.reshape(-1)
        #print(f"{data_view=}")
        idx = replay_buffer.extend(data_view)
        rewards    = data_view["next", "reward"].squeeze(-1)          # [B]
        dones      = data_view["next", "done"].squeeze(-1).float()    # [B]
        q_sa       = data_view["chosen_action_value"].squeeze(-1)     # Q(s,a) [B]
        if len(q_sa.shape) == 2:
            q_sa = q_sa.squeeze(1)

        # 2) compute Q-values at next states
        with torch.no_grad():
            next_obs = data_view["next"]               # [B, ...]
            max_q_next = actor(next_obs)["chosen_action_value"].squeeze(-1)
        # 3) build the TD‐target:  r + γ·maxQ(s',a')·(1−done)
        td_target = rewards + gamma * max_q_next * (1.0 - dones)      # [B]
        # 4) raw TD‐error
        td_error = td_target - q_sa                                    # [B]

        # 5) absolute TD‐error (for PER priorities, etc.)
        abs_td_error = td_error.abs()
        #print(f"td_target: {td_target.shape=}, {abs_td_error.shape=}, {q_sa.shape=}, {rewards.shape=}, {dones.shape=}, {max_q_next.shape=}, {td_error.shape=}")

        # 6) update priorities in the replay buffer
        replay_buffer.update_priority(idx, abs_td_error)
        for _ in range(training_iter_per_batch):
            for _ in range(frames_per_collector // mini_batch_size):
                mini_batch, info = replay_buffer.sample(mini_batch_size, return_info=True)
                mini_batch = mini_batch.to(device)
                loss_vals = loss_module(mini_batch)
                loss_value = loss_vals["loss"]
                dqn_loss_acc += loss_vals["loss"].item()

                # Optimization: backward, grad clipping and optimization step
                optim.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(
                   loss_module.parameters(),
                   max_norm=clip_grad,
                )
                optim.step()
                # optim.zero_grad()
                target_updater.step() # update target network
                #print(f"{mini_batch=}")
                rewards    = mini_batch["next", "reward"].squeeze(-1)          # [B]
                dones      = mini_batch["next", "done"].squeeze(-1).float()    # [B]
                q_sa       = mini_batch["chosen_action_value"].squeeze(-1)     # Q(s,a) [B]
                if len(q_sa.shape) == 2:
                    q_sa = q_sa.squeeze(1)

                # 2) compute Q-values at next states
                with torch.no_grad():
                    next_obs = mini_batch["next"]               # [B, ...]
                    max_q_next = actor(next_obs)["chosen_action_value"].squeeze(-1)
                # 3) build the TD‐target:  r + γ·maxQ(s',a')·(1−done)
                td_target = rewards + gamma * max_q_next * (1.0 - dones)      # [B]
                # 4) raw TD‐error
                td_error = td_target - q_sa                                    # [B]

                # 5) absolute TD‐error (for PER priorities, etc.)
                abs_td_error = td_error.abs()
                #print(f"td_target: {td_target.shape=}, {abs_td_error.shape=}, {q_sa.shape=}, {rewards.shape=}, {dones.shape=}, {max_q_next.shape=}, {td_error.shape=}")


                # 6) update priorities in the replay buffer
                replay_buffer.update_priority(info["index"], abs_td_error)

        logs["collector avg_reward"].append(tensordict_data["next", "reward"].mean().item())
        max_step_count = tensordict_data["step_count"].max().item()

        lines_cleared = tensordict_data["total_lines"].max().item()
        logs["collector lines_cleared"].append(lines_cleared)
        #print(f"MaxStepCount: {max_step_count}")
        logs["collector max step count"].append(max_step_count)
        logs["lr"].append(optim.param_groups[0]["lr"])

        logs["DQN Loss"].append(dqn_loss_acc)
        expl_mod = actor_explore[-1]

        # the current ε is stored in expl_mod.eps
        current_eps = expl_mod.eps.item()      # e.g. 0.7324

        logs["epsilons"].append(current_eps)
        expl_mod.step(frames=frames_per_collector)
       
        if collector_batch % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                env = get_env_func()
                rollouts = []
                for i in range(5):
                    env.reset()
                    rollout = env.rollout(1000, actor)
                    rollouts.append(rollout)
                env.reset()
                eval_rollout = torch.cat(rollouts, dim=0)

                avg_reward = eval_rollout["next", "reward"].mean().item()
                sum_reward = eval_rollout["next", "reward"].sum().item()
                total_steps = eval_rollout["step_count"].sum().item()
                lines_cleared = eval_rollout["next", "total_lines"].max().item()

                
                logs["eval avg reward"].append(avg_reward)
                logs["eval sum reward"].append(sum_reward)
                logs["eval step_count"].append(total_steps)
                logs["eval lines_cleared"].append(lines_cleared)

                
                if sum_reward >= best_model_reward_sum:
                    best_model_reward_sum = sum_reward
                    torch.save(actor.state_dict(), f"{save_dir}best_model_reward_sum.pth")
                    with open(out_file_txt, "a") as f:
                        f.write(f"Best model saved with reward sum {best_model_reward_sum}\n")
                    print(f"Best model saved with reward sum {best_model_reward_sum}")
                if avg_reward >= best_model_reward_avg:
                    best_model_reward_avg = avg_reward
                    torch.save(actor.state_dict(), f"{save_dir}best_model_reward_avg.pth")
                    with open(out_file_txt, "a") as f:
                        f.write(f"Best model saved with reward avg {best_model_reward_avg}\n")
                    print(f"Best model saved with reward avg {best_model_reward_avg}")
                if total_steps >= best_model_step_count:
                    best_model_step_count = total_steps
                    torch.save(actor.state_dict(), f"{save_dir}best_step_count_model.pth")
                    with open(out_file_txt, "a") as f:
                        f.write(f"Best model saved with step count {best_model_step_count}\n")
                    print(f"Best model saved with step count {best_model_step_count}")

                if lines_cleared >= best_model_lines_cleared:
                    best_model_lines_cleared = lines_cleared
                    torch.save(actor.state_dict(), f"{save_dir}best_model_lines_cleared.pth")
                    with open(out_file_txt, "a") as f:
                        f.write(f"Best model saved with lines cleared {best_model_lines_cleared}\n")
                    print(f"Best model saved with lines cleared {best_model_lines_cleared}")


                del eval_rollout
            eval_str = f"Eval {collector_batch//10}/{(total_frames//frames_per_collector)//10}: avg eval reward:{logs['eval avg reward'][-1]}, sum eval reward :{logs['eval sum reward'][-1]}, StepCount:{logs['eval step_count'][-1]}, total_time:{time.time()-start_time:.2f} seconds, Time since last eval: {time.time()-last_eval_time:.2f} seconds"
            last_eval_time = time.time()
            print(eval_str, end='\n', flush=True)
            with open(out_file_txt, "a") as f:
                f.write(eval_str + "\n")


    fig_dir = f"{save_dir}figures/"
    os.makedirs(fig_dir, exist_ok=True)
    graph_logs(logs, fig_dir)



    
    initial_state_dict = {k: v.detach().cpu().clone().to(device) for k, v in actor.state_dict().items()}
    eval_file = f"{save_dir}eval.txt"
    with open(eval_file, "w") as f:
        f.write(f"Best model reward sum: {best_model_reward_sum}\n")
        f.write(f"Best model reward avg: {best_model_reward_avg}\n")
        f.write(f"Best model step count: {best_model_step_count}\n")
        f.write(f"Best model lines cleared: {best_model_lines_cleared}\n")
        f.write("\n\n\n")
    for checkpoint in ["best_model_reward_sum", "best_model_reward_avg", "best_step_count_model", "best_model_lines_cleared"]:
        actor.load_state_dict(torch.load(f"{save_dir}{checkpoint}.pth", map_location=device))
        # assert any(
        #     not torch.allclose(new, initial_state_dict[k], atol=1e-6)
        #     for k, new in actor.state_dict().items()
        # ), "Policy parameters did not update!"
        model_video_dir = f"{save_dir}{checkpoint}/"
        os.makedirs(model_video_dir, exist_ok=True)

        actor.eval()
        td = env.reset()
        for i in range(5):
            env = get_env_func()
            record_video(env, actor, f"{model_video_dir}{i}.mp4")
        lines_cleared = []
        for i in range(32):
            env.reset()
            rollout = env.rollout(1000, actor)
            rollouts.append(rollout)
            lines_cleared.append(rollout["next", "total_lines"].max().item())
        with open(eval_file, "a") as f:
            f.write(f"{checkpoint}: Lines cleared: {sum(lines_cleared)/len(lines_cleared)}\n")
    return
        


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    #train_ppo(get_mcc_env, "MCC", total_frames = 50_000, frames_per_collector = 128)
    # train_dqn(get_mcd_env, "MCd", total_frames = 10_000)

    train_dqn(get_tetris_env, "tetris", total_frames = 10_000)
    train_dqn(get_tetris_env_flat, "tetris_flat", total_frames = 10_000)
