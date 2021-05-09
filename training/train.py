import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from training.model import CnnDQN

# Use GPU, if available
USE_CUDA = torch.cuda.is_available()

class Worker():

    def __init__(self, env, model):
        self.env = env
        self.model = model

    def pass_to_global(self):
        """
        Pass gradients to global network after some timesteps
        """
        raise NotImplementedError



class Trainer():

    def __init__(self, env, parameters):
        self.env = env

        self.load_model()
        self.optimizer = torch.optim.Adam(self.current_model.parameters())
        self.update_target(self.current_model, self.target_model)  # sync nets

        self.num_frames = parameters["num_frames"]
        self.buffersize = parameters["buffersize"]
        self.batch_size = parameters["batch_size"]

        self.gamma = parameters["gamma"]

        self.epsilon_start = parameters["epsilon_start"]
        self.epsilon_final = parameters["epsilon_final"]
        self.epsilon_decay = parameters["epsilon_decay"]

    def load_model(self):
        # try:
        #     if USE_CUDA:
        #         self.current_model = torch.load("training/model.pkl")
        #         self.target_model = torch.load("training/model.pkl")
        #     else:
        #         self.current_model = torch.load(
        #             "training/model.pkl", map_location={'cuda:0': 'cpu'})
        #         self.target_model = torch.load(
        #             "training/model.pkl", map_location={'cuda:0': 'cpu'})
        # except FileNotFoundError:
        self.current_model = CnnDQN(
            self.env.observation_space.shape, self.env.action_space.n)
        self.target_model = CnnDQN(
            self.env.observation_space.shape, self.env.action_space.n)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

    def push_to_buffer(self, *args):
        raise NotImplementedError

    def compute_td_loss(self, *args):
        raise NotImplementedError

    def update_target(self, current_model, target_model):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def plot(self, frame_idx, rewards, losses):
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(122)
        plt.title('loss')
        plt.plot(losses)
        plt.show()

    # def epsilon_by_frame(self, frame_idx):
    #     decay = math.exp(-1. * frame_idx / self.epsilon_decay)
    #     return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * decay

    def epsilon_by_frame(self, frame_idx):
        m = (self.epsilon_final - self.epsilon_start) / self.epsilon_decay
        epsilon_linear = self.epsilon_start + m * frame_idx
        return max(epsilon_linear, self.epsilon_final)

    def train(self):
        # Variables
        state = self.env.reset()
        losses = []
        all_rewards = []
        episode_reward = 0
        games = 0

        # Training
        for frame_idx in range(1, self.num_frames + 1):   
            epsilon = self.epsilon_by_frame(frame_idx)
            # Select action
            action = self.current_model.act(state, epsilon)
            # Move action
            next_state, reward, done, info = self.env.step(action)
            self.push_to_buffer(state, action, reward, next_state, done)
            # Accumulate rewards
            episode_reward += reward
            # Check if game has been terminated
            if done:
                games += 1
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
            else:
                state = next_state
            # Train
            if frame_idx > int(self.buffersize / 2):
                loss = self.compute_td_loss(self.batch_size, frame_idx)
                losses.append(loss.data.item())
            # Update Target
            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)
                loss = round(np.mean(losses[-500:]), 5)
                print("{} frames: {} games, {} reward, {} max reward, {} loss".format(
                    frame_idx, games, round(np.mean(all_rewards[-5:]), 5), 
                    round(np.max(all_rewards), 4), loss
                ))
            # Save the current model
            if frame_idx % 10000 == 0:
                torch.save(self.current_model, "training/model.pkl")
        else:
            all_rewards.append(episode_reward)

        torch.save(self.current_model, "training/model.pkl")
        print('Training finished.')

        self.plot(frame_idx, all_rewards, losses)
