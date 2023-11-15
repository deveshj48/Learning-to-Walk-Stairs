from training import run_experiment, render_policy
from environment import *


example_config = {
    "experiment_name": "test_CustomAnt",
    "env": "CustomAnt-v0",
    "env": "CartPole-v1",
    "n_sessions": 50,
    "env_steps": 1000000, 
    "population_size": 256,
    "learning_rate": 0.06,
    "noise_std": 0.1,
    "noise_decay": 0.99, # optional
    "lr_decay": 1.0, # optional
    "decay_step": 20, # optional
    "eval_step": 10, 
    "hidden_sizes": (40, 40)
  }

policy = run_experiment(example_config, n_jobs=4, verbose=True)

# to render policy perfomance
render_policy('model_policy.pkl', example_config["env"], n_videos=10)