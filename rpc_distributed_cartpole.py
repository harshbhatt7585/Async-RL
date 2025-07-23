import os
import argparse
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributed.rpc import RRef, rpc_async, remote

GAMMA = 0.99
MAX_EPISODES = 50

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.dropout(self.affine1(x)))
        return F.softmax(self.affine2(x), dim=1)

class Observer:
    def __init__(self):
        self.env = gym.make("CartPole-v1")

    def run_episode(self, agent_rref: RRef):
        state, _ = self.env.reset()
        ep_reward = 0
        for _ in range(1000):
            # ask the Agent for an action
            action = agent_rref.rpc_sync().select_action(state)
            state, reward, done, _, _ = self.env.step(action)
            agent_rref.rpc_sync().report_reward(reward)
            ep_reward += reward
            if done:
                break
        return ep_reward

class Agent:
    def __init__(self, observer_name: str):
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.saved_log_probs = []
        self.rewards = []
        # create a remote Observer on the other node
        self.ob_rref = remote(observer_name, Observer)
        self.self_rref = RRef(self)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def report_reward(self, reward):
        self.rewards.append(reward)

    def run_episode(self):
        # invoke Observer.run_episode(agent_rref) remotely
        fut = rpc_async(self.ob_rref.owner(),
                        Observer.run_episode,
                        args=(self.ob_rref, self.self_rref))
        return fut.wait()

    def finish_episode(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = [-log_prob * R for log_prob, R in zip(self.saved_log_probs, returns)]
        self.optimizer.zero_grad()
        torch.cat(policy_loss).sum().backward()
        self.optimizer.step()

        self.saved_log_probs.clear()
        self.rewards.clear()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_rank", type=int, required=True,
                        help="Node rank (0 for agent node, 1 for observer node)")
    parser.add_argument("--world_size", type=int, default=2,
                        help="Total number of nodes")
    parser.add_argument("--master_addr", type=str, required=True,
                        help="Address of the master (agent) node")
    parser.add_argument("--master_port", type=str, default="29500",
                        help="Port for rendezvous")
    args = parser.parse_args()

    # Set up environment for torchrun rendezvous
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    # Name each RPC worker uniquely
    if args.node_rank == 0:
        worker_name = "agent"
    else:
        worker_name = "observer"

    rpc.init_rpc(
        name=worker_name,
        rank=args.node_rank,
        world_size=args.world_size,
    )

    if args.node_rank == 0:
        agent = Agent(observer_name="observer")
        for ep in range(1, MAX_EPISODES + 1):
            total_reward = agent.run_episode()
            agent.finish_episode()
            print(f"[Episode {ep}] Total Reward: {total_reward}")
        # after training, shut down
        rpc.shutdown()
    else:
        # observer just initializes and waits for RPC calls
        rpc.shutdown()

if __name__ == "__main__":
    main()
