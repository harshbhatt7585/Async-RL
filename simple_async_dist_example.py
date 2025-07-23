import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import gym


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.policy(x), self.value(x)


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    return local_rank, global_rank


def rollout(env, local_model, t_max, device):
    state, _ = env.reset()
    values, log_probs, rewards, entropies = [], [], [], []
    episode_reward = 0.0

    for _ in range(t_max):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = local_model(state_tensor)
            probs = F.softmax(logits, dim=-1)
            log_probs_dist = F.log_softmax(logits, dim=-1)
            dist_ = torch.distributions.Categorical(probs)
            action = dist_.sample()

        next_state, reward, done, _, _ = env.step(action.item())

        values.append(value)
        log_probs.append(log_probs_dist.gather(1, action.unsqueeze(0)))
        rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
        entropies.append(-(log_probs_dist * probs).sum(1, keepdim=True))

        episode_reward += reward
        state = next_state

        if done:
            state, _ = env.reset()
            break

    return values, log_probs, rewards, entropies, episode_reward


def average_gradients(model):
    # Gradient averaging across all processes
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()


def train_worker(global_model, env_name, t_max=20, gamma=0.99, max_episodes=1000, sync_interval=10):
    local_rank, global_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    global_model.to(device)
    global_model = DDP(global_model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)
    env = gym.make(env_name)

    episode = 0
    while episode < max_episodes:
        values, log_probs, rewards, entropies, episode_reward = rollout(env, global_model.module, t_max, device)

        # Bootstrap value
        with torch.no_grad():
            state_tensor = torch.tensor(env.reset()[0], dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value = global_model.module(state_tensor)
            R = next_value.detach()

        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.cat(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()

        # Periodic sync across workers
        if episode % sync_interval == 0:
            average_gradients(global_model)
        optimizer.step()

        if global_rank == 0 and episode % 10 == 0:
            print(f"[Rank {global_rank}] Episode {episode} | Reward: {episode_reward:.2f} | Loss: {total_loss.item():.3f}")

        episode += 1

    env.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    env.close()

    train_worker(model, env_name=args.env_name)
