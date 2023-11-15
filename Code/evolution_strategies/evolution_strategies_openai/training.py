import gym
import pickle
import uuid

import numpy as np

from tqdm import tqdm
# from joblib import Parallel
from collections import defaultdict

from gym import wrappers

from linear import ThreeLayerNetwork
from es import OpenAiES
from plot import plot_rewards
from evaluation import eval_policy_delayed, eval_policy
from environment import *

# env: (n_states, n_actions)
ENV_INFO = {
    "CustomAnt-v0": (27, 8),
    "CartPole-v1":(4,2)

}


def train_loop(policy, env, config, n_jobs=1, verbose=True):
    es = OpenAiES(
        model=policy, 
        learning_rate=config["learning_rate"], 
        noise_std=config["noise_std"],
        noise_decay=config.get("noise_decay", 1.0),
        lr_decay=config.get("lr_decay", 1.0),
        decay_step=config.get("decay_step", 50)
    )
    
    log = defaultdict(list)
    for session in tqdm(range(config["n_sessions"])):
        population = es.generate_population(config["population_size"])

        # rewards = (eval_policy(new_policy, env, config["env_steps"]) for new_policy in population)
        rewards = []
        for new_policy in population:
            rewards.append(eval_policy(new_policy, env, config["env_steps"]))
        # rewards_jobs = (eval_policy_delayed(new_policy, env, config["env_steps"]) for new_policy in population) # for parallel
        # rewards = np.array(Parallel(n_jobs=n_jobs)(rewards_jobs))

        # print(list(rewards_jobs)[0][0])
        rewards = np.array(rewards)
        # print(rewards.size)  
        
        if (rewards.size>0):
            es.update_population(rewards)

        # populations stats
        log["pop_mean_rewards"].append(np.mean(rewards))
        log["pop_std_rewards"].append(np.std(rewards))
        
        # best policy stats
        if session % config.get("eval_step", 2) == 0:
            best_policy = es.get_model()

            best_rewards = np.zeros(10)
            for i in range(10):
                best_rewards[i] = eval_policy(best_policy, env, config["env_steps"])

            if verbose:
                # TODO: add timestamp
                print(f"Session: {session}")   
                print(f"Mean reward: {round(np.mean(rewards), 4)}", f"std: {round(np.std(rewards), 3)}")
                print(f"lr: {round(es.lr, 5)}, noise_std: {round(es.noise_std, 5)}")

            log["best_mean_rewards"].append(np.mean(best_rewards))
            log["best_std_rewards"].append(np.std(best_rewards))            

    return log


def run_experiment(config, n_jobs=4, verbose=True):
    env = gym.make(config["env"])

    n_states, n_actions = ENV_INFO[config["env"]]

    if config.get("init_model", None):
        policy = ThreeLayerNetwork.from_model(config["init_model"])
        
        assert policy.in_features == n_states, "not correct policy input dims"
        assert policy.out_features == n_actions, "not correct policy output dims"
    else:
        policy = ThreeLayerNetwork(
            in_features=n_states, 
            out_features=n_actions, 
            hidden_sizes=config["hidden_sizes"]
        )
    log = train_loop(policy, env, config, n_jobs, verbose)

    # if config.get("log_path", None):
    with open("model_log.pkl", "wb") as file:
        pickle.dump(log, file)

    # if config.get("model_path", None):
    with open("model_policy.pkl", "wb") as file:
        pickle.dump(policy, file)



    plot_rewards(log["best_mean_rewards"], log["best_std_rewards"], config)

    return policy


def render_policy(model_path, env_name, n_videos=1):
    with open(model_path, "rb") as file:
        policy = pickle.load(file)

    # model_name = model_path.split("/")[-1].split(".")[0]
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, 'video', lambda x: x % 2 == 0)
    for i in range(n_videos):
        # env = gym.make(env_name)
        # env = wrappers.Monitor(env, f'videos/{model_name}/' + str(uuid.uuid4()), force=True)



        print(eval_policy(policy, env, n_steps=1600))
        env.close()


if __name__ == "__main__":
    # TODO: analyse population stat from logs
    render_policy("model_policy.pkl", "CartPole-v1")

