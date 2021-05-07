from dinoEnv import DinoEnv
from training import DQN

# Training parameters
parameters = {
    "num_frames" : 1000000,
    "batch_size" : 32,
    "gamma" : 0.95,

    "buffersize" : 20000,

    "epsilon_start" : 0.5,
    "epsilon_final" : 0.01,
    "epsilon_decay" : 500000,
}

env = DinoEnv()
trainer = DQN(env, parameters)

trainer.train()
