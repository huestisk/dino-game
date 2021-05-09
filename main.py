from dinoEnv import DinoEnv
from training import PriorDQN

# Training parameters
parameters = {
    "num_frames" : 2000000,
    "batch_size" : 32,
    "gamma" : 0.99,

    "buffersize" : 20000,

    "epsilon_start" : 0.8,
    "epsilon_final" : 0.01,
    "epsilon_decay" : 1500000,

    "alpha" : 1.0,
    "beta_start" : 0.4,
    "beta_frames" : 10000
}

env = DinoEnv()
trainer = PriorDQN(env, parameters)

trainer.train()
