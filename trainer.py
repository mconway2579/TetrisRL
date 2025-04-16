import torch
from torchrl.envs import GymWrapper
from TetrisEnv import TetrisEnv  # wherever you put the class


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.backends.mps.device if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    # instantiate your gym‐style env
    base_env = TetrisEnv()

    # wrap it in TorchRL
    # by default this will:
    #  • convert obs/rewards/dones into a tensordict
    #  • convert your discrete actions into a torch.Tensor
    #  • live on CPU
    env = GymWrapper(
        base_env,
        device=device,            # where tensors should live
        categorical_action_encoding=False,     # discrete → OneHot / Categorical spec
        from_pixels=False,                     # we already return raw pixel obs
    )

    # reset returns a TensorDict
    td = env.reset()
    print(td)
    # TensorDict(
    #    fields={
    #      action: …,
    #      next: TensorDict(fields={ … observation, reward, done … }),
    #    },
    #    batch_size=torch.Size([]), device=cpu
    # )

    # sample a random action and step
    td = env.rand_step()  # does both: samples action, steps, returns next tensordict
    print(td)
