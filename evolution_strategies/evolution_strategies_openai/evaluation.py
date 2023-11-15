import numpy as np
from joblib import delayed

def eval_policy(policy, env, n_steps=200):
    
    total_reward = 0
    
    obs = env.reset()[0]
    for i in range(n_steps):
        action = policy.predict(np.array(obs).reshape(1, -1), scale="tanh") # for continuous spaces/

        new_obs, reward, terminated, truncated, info = env.step(action)        
        total_reward = total_reward + reward
        obs = new_obs

        if terminated or truncated:
            break

    return total_reward


# for parallel
# eval_policy_delayed = delayed(eval_policy)
eval_policy_delayed = eval_policy