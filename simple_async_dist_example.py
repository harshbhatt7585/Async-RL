import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
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


# Global model container for RPC
class GlobalModel:
    def __init__(self, obs_dim, act_dim):
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def apply_gradients(self, grads):
        for p, g in zip(self.model.parameters(), grads):
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}


def remote_apply_gradients(grads):
    return GLOBAL_MODEL.apply_gradients(grads)


def remote_get_weights():
    return GLOBAL_MODEL.get_weights()


def rollout_worker(rank, world_size, env_name, t_max, gamma, max_episodes):
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    local_model = ActorCritic(obs_dim, act_dim)

    for episode in range(max_episodes):
        # Pull latest weights
        state_dict = rpc.rpc_sync("ps", remote_get_weights, args=())
        local_model.load_state_dict(state_dict)

        values, log_probs, rewards, entropies = [], [], [], []
        state, _ = env.reset()

        for _ in range(t_max):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = local_model(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(action.item())

            values.append(value)
            log_probs.append(torch.log(probs[0, action]))
            rewards.append(reward)
            entropies.append(-(probs * probs.log()).sum())

            state = next_state
            if done:
                break

        R = torch.zeros(1, 1)
        returns = []
        for r in reversed(rewards):
            R = torch.tensor([[r]]) + gamma * R
            returns.insert(0, R)

        values = torch.cat(values)
        log_probs = torch.stack(log_probs).unsqueeze(1)
        entropies = torch.stack(entropies).unsqueeze(1)
        returns = torch.cat(returns).detach()

        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # Compute gradients
        local_model.zero_grad()
        total_loss.backward()
        grads = [p.grad for p in local_model.parameters()]

        # Push to global model
        rpc.rpc_sync("ps", remote_apply_gradients, args=(grads,))

        if episode % 10 == 0:
            print(f"Worker {rank} | Episode {episode} | Reward: {sum(rewards):.2f}")

    rpc.shutdown()


def run_rpc(env_name, world_size, t_max=20, gamma=0.99, max_episodes=200):
    mp.set_start_method("spawn", force=True)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    global GLOBAL_MODEL
    GLOBAL_MODEL = GlobalModel(obs_dim, act_dim)

    rpc.init_rpc("ps", rank=0, world_size=world_size)

    processes = []
    for rank in range(1, world_size):
        p = mp.Process(target=rollout_worker, args=(rank, world_size, env_name, t_max, gamma, max_episodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    rpc.shutdown()


if __name__ == "__main__":
    run_rpc(env_name="CartPole-v1", world_size=3)  # 1 PS + 2 Workers
