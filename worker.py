# worker.py - Each worker creates its own optimizer
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from model import ActorCritic

def worker_fn(global_model, global_optimizer, env_name, worker_id, gamma=0.99, t_max=20):
    torch.manual_seed(worker_id + 1000)  # Different seed per worker
    np.random.seed(worker_id + 1000)
    
    env = gym.make(env_name)
    env.reset(seed=worker_id + 1000)  # Different environment seed per worker
    
    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    local_model.load_state_dict(global_model.state_dict())

    state, _ = env.reset()
    done = False
    episode = 0
    step_count = 0

    while True:
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # Sync with global model at start of each trajectory
        local_model.load_state_dict(global_model.state_dict())

        for _ in range(t_max):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = local_model(state_tensor)
            
            # Add small epsilon for numerical stability
            prob = F.softmax(logits, dim=-1)
            log_prob_dist = F.log_softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(int(action.item()))

            # Store values
            log_probs.append(log_prob_dist.gather(1, action.unsqueeze(0)))
            values.append(value)
            rewards.append(reward)
            # Calculate entropy for exploration
            entropies.append(-(log_prob_dist * prob).sum(1))

            state = next_state
            step_count += 1
            
            if done:
                state, _ = env.reset()
                episode += 1
                break

        # Bootstrap value
        if done:
            R = torch.zeros(1, 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, value = local_model(state_tensor)
            R = value.detach()

        # Calculate returns and advantages
        returns = []
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        # Calculate advantages
        advantages = returns - values
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        loss.backward()
        

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param._grad = local_param.grad.clone()
            
        
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=5.0)
        global_optimizer.step()
        global_optimizer.zero_grad()

        # Print from all workers
        if episode > 0 and episode % 10 == 0:
            print(f"[Worker {worker_id}] Episode: {episode}, Loss: {loss.item():.3f}, Steps: {step_count}")
            step_count = 0