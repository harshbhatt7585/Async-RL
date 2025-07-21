# main.py - Simpler approach without SharedAdam
import torch
import torch.multiprocessing as mp
import gym
from model import ActorCritic
from worker import worker_fn

def main():
    mp.set_start_method("spawn") 

    env_name = "CartPole-v1"
    dummy_env = gym.make(env_name)
    input_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    global_model = ActorCritic(input_dim, action_dim)
    global_model.share_memory()  

    # Simple Adam optimizer - each worker will create its own
    # We'll pass the learning rate instead
    lr = 1e-4

    num_workers = 4
    processes = []

    for worker_id in range(num_workers):
        p = mp.Process(target=worker_fn, args=(global_model, lr, env_name, worker_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()