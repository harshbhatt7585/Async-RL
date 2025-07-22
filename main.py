import torch
import torch.multiprocessing as mp
import gym
import wandb
import numpy as np
from collections import deque
import time
from model import ActorCritic
from worker import worker_fn

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def setup_wandb(config):
    """Initialize wandb for experiment tracking."""
    pass

def metrics_logger(metric_queue, num_workers, config, log_interval=100):
    """
    Separate process to handle metrics logging to wandb.
    This prevents blocking the training workers.
    """
    wandb.init(
        project="a3c-cartpole",
        config=config,
        name=f"a3c-{config['num_workers']}workers-lr{config['lr']}"
    )
    
    worker_metrics = {i: {
        'rewards': deque(maxlen=1000),
        'losses': deque(maxlen=1000),
        'entropies': deque(maxlen=1000),
        'policy_losses': deque(maxlen=1000),
        'value_losses': deque(maxlen=1000)
    } for i in range(num_workers)}
    
    step_count = 0
    last_log_time = time.time()
    
    while True:
        try:
            # Non-blocking get with timeout
            data = metric_queue.get(timeout=1.0)
            
            if data is None:  # Shutdown signal
                break
                
            worker_id, metric_type, value = data
            worker_metrics[worker_id][metric_type].append(value)
            step_count += 1
            
            # Log to wandb periodically
            current_time = time.time()
            if step_count % log_interval == 0 or current_time - last_log_time > 30: 
                log_aggregated_metrics(worker_metrics, step_count)
                last_log_time = current_time
                
        except:
            continue
    
    # Final logging
    log_aggregated_metrics(worker_metrics, step_count)
    wandb.finish()

def log_aggregated_metrics(worker_metrics, step):
    """Log aggregated metrics to wandb."""
    all_rewards = []
    all_losses = []
    all_entropies = []
    worker_stats = {}
    
    for worker_id, metrics in worker_metrics.items():
        rewards = list(metrics['rewards'])
        losses = list(metrics['losses'])
        entropies = list(metrics['entropies'])
        
        if rewards:
            all_rewards.extend(rewards)
            worker_stats[f'worker_{worker_id}/avg_reward'] = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
            worker_stats[f'worker_{worker_id}/latest_reward'] = rewards[-1]
        
        if losses:
            all_losses.extend(losses)
            worker_stats[f'worker_{worker_id}/avg_loss'] = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
        
        if entropies:
            all_entropies.extend(entropies)
            worker_stats[f'worker_{worker_id}/avg_entropy'] = np.mean(entropies[-10:]) if len(entropies) >= 10 else np.mean(entropies)
    
    log_data = {'global_step': step}
    
    if all_rewards:
        log_data.update({
            'global/avg_reward': np.mean(all_rewards),
            'global/max_reward': np.max(all_rewards),
            'global/min_reward': np.min(all_rewards),
            'total_episodes': len(all_rewards),
        })
    
    if all_losses:
        log_data.update({
            'globa/avg_loss': np.mean(all_losses),
            'global/max_loss': np.max(all_losses),
            'global/min_loss': np.min(all_losses),
        })
    
    if all_entropies:
        log_data.update({
            'global/avg_entropy': np.mean(all_entropies),
        })
    
    log_data.update(worker_stats)
    wandb.log(log_data)

def main():
    mp.set_start_method("spawn")

    env_name = "CartPole-v1"
    dummy_env = gym.make(env_name)
    input_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    # Configuration
    config = {
        'env_name': env_name,
        'input_dim': input_dim,
        'action_dim': action_dim,
        'lr': 1e-4,
        'num_workers': 4,
        'algorithm': 'A3C',
        'max_episodes': 1000
    }
    

    global_model = ActorCritic(input_dim, action_dim)
    global_model.share_memory()

    global_optimizer = SharedAdam(global_model.parameters(), lr=config['lr'])

    # Use multiprocessing Queue instead of manager.dict - much faster!
    metric_queue = mp.Queue(maxsize=10000)  # Large buffer to prevent blocking
    
    # Start metrics logger process
    logger_process = mp.Process(
        target=metrics_logger, 
        args=(metric_queue, config['num_workers'], config)
    )
    logger_process.start()

    # Start worker processes
    processes = []
    for worker_id in range(config['num_workers']):
        p = mp.Process(
            target=worker_fn,
            args=(global_model, global_optimizer, env_name, worker_id, metric_queue, config['max_episodes'])
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    metric_queue.put(None)
    logger_process.join()
        
    print("Training completed! Check your wandb dashboard for visualizations.")

if __name__ == "__main__":
    main()