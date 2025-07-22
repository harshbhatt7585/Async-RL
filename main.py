import torch
import torch.multiprocessing as mp
import gym
import matplotlib.pyplot as plt
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


def plot_metrics(metrics, num_workers):
    """
    Plot each worker's rewards in a single figure with subplots.
    """
    # Create a vertical grid of subplots
    fig, axes = plt.subplots(nrows=num_workers, ncols=1, figsize=(8, 4 * num_workers), sharex=True)
    # If only one worker, axes is not an array
    if num_workers == 1:
        axes = [axes]

    for wid, ax in enumerate(axes):
        rewards = list(metrics[wid])
        if rewards:
            ax.plot(rewards)
        ax.set_title(f'Worker {wid} Episode Rewards')
        ax.set_ylabel('Reward')
        ax.grid(True)

    axes[-1].set_xlabel('Episode')
    fig.tight_layout()
    plt.show()


def main():
    mp.set_start_method("spawn")

    env_name = "CartPole-v1"
    dummy_env = gym.make(env_name)
    input_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    global_model = ActorCritic(input_dim, action_dim)
    global_model.share_memory()

    lr = 1e-4
    global_optimizer = SharedAdam(global_model.parameters(), lr=lr)

    num_workers = 4
    manager = mp.Manager()
    metrics = manager.dict({i: manager.list() for i in range(num_workers)})

    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_fn,
            args=(global_model, global_optimizer, env_name, worker_id, metrics)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    plot_metrics(metrics, num_workers)


if __name__ == "__main__":
    main()
