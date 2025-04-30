from data_collector import get_collecter, get_replay_buffer
from utils import select_device, record_video, graph_logs
from Enviorments import get_tetris_env, get_mcd_env
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import cv2
import os
import numpy as np

# for dqn
from torchrl.objectives import DQNLoss, SoftUpdate

from networks import get_EGDQN  # Use get_EGDQN from the second code snippet

#https://pytorch.org/rl/main/tutorials/coding_ppo.html#policy


def train_dqn(get_env_func, env_name, lr=1e-4, frames_per_collector=256, total_frames=1_000_000, batches_to_store=1024, mini_batch_size=128, training_iter_per_batch=10, gamma=0.999, entropy_eps=2, clip_grad=1, eps_start=1.0, eps_end=0.05):
    n_batches = np.ceil(total_frames / frames_per_collector)
    n_batches = np.ceil(n_batches / 10) * 10
    total_frames = (n_batches * frames_per_collector) +1

    save_dir = f"results/dqn_{env_name}_{lr=}_{total_frames=}_{mini_batch_size=}_{training_iter_per_batch=}_{entropy_eps=}/"
    os.makedirs(save_dir, exist_ok=True)
    device = select_device()

    actor, actor_explore = get_EGDQN(get_env_func, eps_start, eps_end, total_frames)
    actor.to(device)
    actor_explore.to(device)
    collector = get_collecter(get_env_func, actor, frames_per_collector, total_frames)
    # print(f"Collector: {collector}")
    # print(f"{dir(collector)=}")

    # print(f"Collector: {collector.exploration_type}")
    # input()
    replay_buffer = get_replay_buffer(batches_to_store, frames_per_collector, mini_batch_size)
    
    # Define DQN loss
    env = get_env_func()
    loss_module = DQNLoss(actor, action_space=env.action_spec).to(device)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, tau=0.995) #soft update of the target network
    
    # clip_epsilon = 0.2
    # """loss_module = PPOLoss(
    #     actor_network=ppo_policy,
    #     critic_network=value_module,
    #     entropy_bonus=True,
    #     entropy_coef=entropy_eps,
    #     critic_coef=critic_coef,
    #     loss_critic_type="smooth_l1",
    # ).to(device)"""
    # loss_module = ClipPPOLoss(
    #     actor_network=actor,
    #     critic_network=actor_explore,
    #     clip_epsilon=clip_epsilon,
    #     entropy_bonus=bool(entropy_eps),
    #     entropy_coef=entropy_eps,
    #     # these keys match by default but we set this for completeness
    #     critic_coef=critic_coef,
    #     loss_critic_type="smooth_l1",
    #     normalize_advantage=True,
    #     value_clip=True
    # )
   

    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_collector, 0.0
    )

    #batch_td = next(iter(collector))
    #print(f"Batch: {batch_td.shape}")
    #batch_td = batch_td.reshape(-1)
    #print(f"Batch: {batch_td.shape}")



    # advantage_module = GAE(
    #     gamma=gamma, lmbda=lmbda, value_network=actor_explore, average_gae=True
    # )

    logs = defaultdict(list)
    out_file_txt = f"{save_dir}training.txt"
    with open(out_file_txt, "w") as f:
        f.write(save_dir + "\n")
    best_model = None
    best_model_score = -np.inf
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for collector_batch, tensordict_data in enumerate(collector):
        if len(tensordict_data["action"].shape) == 2 and tensordict_data["action"].shape[1] == 1:
            tensordict_data["action"] = tensordict_data["action"].squeeze(1)
        # we now have a batch of data to work with. Let's learn something from it.
        objective_loss_acc = 0.0
        # critic_loss_acc = 0.0
        # entropy_loss_acc = 0.0
        tensordict_data["next", "reward"] *= collector_batch/n_batches
        #print(f"{logs['actions']}")
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(training_iter_per_batch):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            # advantage_module(tensordict_data)
            for _ in range(frames_per_collector // mini_batch_size):
                mini_batch = replay_buffer.sample(mini_batch_size)
                #print(f"mini_batch: {mini_batch}")
                #input("Press enter to continue")
                loss_vals = loss_module(mini_batch.to(device))
                #print(f"loss_vals: {loss_vals}")
                #print(f"loss_vals: {loss_vals}")
                loss_value = (
                    loss_vals["loss"]  
                    # + loss_vals["loss_critic"]   KeyError: 'key "loss_critic" not found in TensorDict with keys [\'loss\']'
                    # + loss_vals["loss_entropy"]
                )
                objective_loss_acc += loss_vals["loss"].item()
                # critic_loss_acc += loss_vals["loss_critic"].item()
                # entropy_loss_acc += loss_vals["loss_entropy"].item()

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
                # optim.zero_grad()
                target_updater.step() # update target network

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        max_step_count = tensordict_data["step_count"].max().item()
        #print(f"MaxStepCount: {max_step_count}")
        logs["step_count"].append(max_step_count)
        logs["lr"].append(optim.param_groups[0]["lr"])

        logs["loss objective"].append(objective_loss_acc)
        expl_mod = actor_explore[-1]

        # the current ε is stored in expl_mod.eps
        current_eps = expl_mod.eps.item()      # e.g. 0.7324

        logs["epsilons"].append(current_eps)
        expl_mod.step(frames=frames_per_collector)
        # logs["loss critic"].append(critic_loss_acc)
        # logs["loss entropy"].append(entropy_loss_acc)
        
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
                eval_rollout = env.rollout(1000, actor)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                if env_name == "tetris":
                    total_reward =  eval_rollout["step_count"].max().item()
                else:
                    total_reward =  eval_rollout["next", "reward"].sum().item()
                logs["eval reward (sum)"].append(
                   total_reward
                )
                if total_reward > best_model_score:
                    best_model_score = total_reward
                    best_model = actor.state_dict()
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
    graph_logs(logs, save_dir)


    model_video_dir = f"{save_dir}model_video/"
    os.makedirs(model_video_dir, exist_ok=True)

    actor.load_state_dict(torch.load(f"{save_dir}best_model.pth", map_location=device))
    actor.eval()
    td = env.reset() # why we need to reset the env?
    for i in range(5):
        env = get_env_func()
        record_video(env, actor, f"{model_video_dir}{i}.mp4")
    return
        


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    #train_ppo(get_mcc_env, "MCC", total_frames = 50_000, frames_per_collector = 128)
    train_dqn(get_mcd_env, "MCd", total_frames = 10_000)

    train_dqn(get_tetris_env, "tetris", total_frames = 10_000)