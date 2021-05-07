import gym
import torch
import random
import numpy as np
from collections import deque
from training.train import Trainer

# Use GPU, if available
USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x

## From https://github.com/higgsfield/RL-Adventure/blob/master/common/wrappers.py
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(Trainer):

    def __init__(self, env, parameters):
        super(DQN, self).__init__(env, parameters)
        self.replay_buffer = ReplayBuffer(self.buffersize)

    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def compute_td_loss(self, batch_size, *args):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.current_model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)
        next_q_value = next_q_state_values.gather(
            1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).abs()
        loss[loss.le(10)] = loss[loss.le(10)].pow(2)
        loss[loss.gt(10)] = (loss[loss.gt(10)] + 95) / 2
        loss[loss.gt(150)] = 150
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
