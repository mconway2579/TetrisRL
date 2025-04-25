from data_collector import get_collecter, get_replay_buffer
from networks import get_PPO_policy
from utils import select_device
from torchrl.objectives import PPOLoss
from Enviorments import get_tetris_env, get_mc_env
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from torchrl.objectives.value import GAE
from data_collector import FRAMES_PER_COLLECTER as frames_per_batch
from data_collector import REPLAY_BUFFER_BATCH_SIZE as sub_batch_size
from data_collector import TOTAL_FRAMES
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import cv2
#https://pytorch.org/rl/main/tutorials/coding_ppo.html#policy

def train(get_env_func):
    
    device = select_device()
    lr = 1e-6
    max_grad_norm = 1.0
    
    EPOCHS_PER_BATCH=10

    
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    ppo_policy, value_module = get_PPO_policy(get_env_func)
    ppo_policy.to(device)
    value_module.to(device)
    collector = get_collecter(get_env_func, ppo_policy)
    replay_buffer = get_replay_buffer()


    loss_module = PPOLoss(
        actor_network=ppo_policy,
        critic_network=value_module,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    ).to(device)


    optim = torch.optim.Adam(list(ppo_policy.parameters()) + list(value_module.parameters()), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, TOTAL_FRAMES // frames_per_batch, 0.0
    # )
    #batch_td = next(iter(collector))
    #print(f"Batch: {batch_td.shape}")
    #batch_td = batch_td.reshape(-1)
    #print(f"Batch: {batch_td.shape}")





    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    logs = defaultdict(list)
    pbar = tqdm(total=TOTAL_FRAMES)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(EPOCHS_PER_BATCH):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            replay_buffer.extend(tensordict_data)

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                #print(f"Subdata: {subdata}")
                #input("Press enter to continue")
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
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
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        # scheduler.step()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show(block = True)


    env = get_env_func()
    td = env.reset()
    acc = 0
    while True:
        try:
            rgb_array = env.render()
            cv2.imshow("Game", rgb_array)
            cv2.waitKey(1)
            td = ppo_policy(td)
            td = env.step(td)
            if td["next", "done"]:
                td = env.reset()
                acc +=1
            if acc > 1:
                break
        except KeyboardInterrupt:
            break
    return
        


if __name__ == "__main__":
    train(get_mc_env)
    train(get_tetris_env)