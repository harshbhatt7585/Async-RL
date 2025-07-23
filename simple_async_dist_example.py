import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value




# collect experience
def rollout(env, local_model, t_max, device='cpu'):
    state, _ = env.reset()
    
    values, log_probs, rewards, entropies = [], [], [], []
    episode_reward = 0.0

    for _ in range(t_max):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = local_model(state_tensor)
            probs = F.softmax(logits, dim=-1)
            log_probs_dist = F.log_softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

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

    return {
        "values": values,
        "log_probs": log_probs,
        "rewards": rewards,
        "entropies": entropies,
        "episode_reward": episode_reward,
    }
    
            
    