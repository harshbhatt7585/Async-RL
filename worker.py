import torch
import torch.nn.functional as F
import gym
import numpy as np
from model import ActorCritic
from gym.wrappers import RecordVideo, TimeLimit
import os


def worker_fn(global_model, global_optimizer, env_name, worker_id, metric_queue, max_episodes=1000, gamma=0.99, t_max=20):
    video_folder = f"./videos/worker_{worker_id}"
    os.makedirs(video_folder, exist_ok=True)

    torch.manual_seed(worker_id + 1000)
    np.random.seed(worker_id + 1000)
    env = gym.make(env_name, render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: ep >= 200 and ep % 50 == 0,
        name_prefix=f"worker{worker_id}"
    )

    env.reset(seed=worker_id + 1000)

    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    state, _ = env.reset()
    done = False
    episode = 0
    step_count = 0

    while episode < max_episodes:
        # Copy weights from global model
        local_model.load_state_dict(global_model.state_dict())

        values, log_probs, rewards, entropies = [], [], [], []
        episode_reward = 0.0

        # Collect t_max steps of experience
        for _ in range(t_max):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = local_model(state_tensor)
            prob = F.softmax(logits, dim=-1)
            log_prob_dist = F.log_softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(int(action.item()))
            episode_reward += reward

            values.append(value)
            log_probs.append(log_prob_dist.gather(1, action.unsqueeze(0)))
            rewards.append(reward)
            entropies.append(-(log_prob_dist * prob).sum(1, keepdim=True))

            state = next_state
            step_count += 1

            if done:
                # Send episode reward to metrics queue (non-blocking)
                try:
                    metric_queue.put((worker_id, 'rewards', episode_reward), block=False)
                except:
                    pass  # Queue full, skip this metric
                
                # Reset for next episode
                state, _ = env.reset()
                episode += 1
                episode_reward = 0.0
                break

        # Bootstrap value for next state (if not terminal)
        if done:
            R = torch.zeros(1, 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, value = local_model(state_tensor)
            R = value.detach()

        # Compute returns using discounted rewards
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        # Convert to tensors
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        # Compute advantages
        advantages = returns - values
        
        # Normalize advantages (only if we have multiple samples)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # Send metrics to queue (non-blocking to avoid slowing down training)
        try:
            metric_queue.put((worker_id, 'losses', total_loss.item()), block=False)
            metric_queue.put((worker_id, 'entropies', -entropy_loss.item()), block=False)
            metric_queue.put((worker_id, 'policy_losses', policy_loss.item()), block=False)
            metric_queue.put((worker_id, 'value_losses', value_loss.item()), block=False)
        except:
            pass  # Queue full, skip these metrics

        # Zero gradients
        global_optimizer.zero_grad()
        
        # Backpropagate
        total_loss.backward()
        
        # Copy gradients from local model to global model
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param._grad is not None:
                global_param._grad += local_param.grad
            else:
                global_param._grad = local_param.grad.clone()

        # Clip gradients and update global model
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=5.0)
        global_optimizer.step()

        # Print progress (less frequent to reduce overhead)
        if episode > 0 and episode % 50 == 0:
            print(f"[Worker {worker_id}] Episode: {episode}, Latest Loss: {total_loss.item():.3f}")

    env.close()