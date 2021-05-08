import torch
import numpy as np
from training.train import Trainer

# Use GPU, if available
USE_CUDA = torch.cuda.is_available()
def Variable(x): return x.cuda() if USE_CUDA else x

# From https://github.com/higgsfield/RL-Adventure/blob/master


class PrioritizedBuffer(object):

    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*samples)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class PriorDQN(Trainer):

    def __init__(self, env, parameters):
        super(PriorDQN, self).__init__(env, parameters)

        self.replay_buffer = PrioritizedBuffer(
            self.buffersize, parameters["alpha"])

        self.beta_start = parameters["beta_start"]
        self.beta_frames = parameters["beta_frames"]

    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def beta_by_frame(self, frame_idx):
        beta = self.beta_start + frame_idx * \
            (1.0 - self.beta_start) / self.beta_frames
        return min(1.0, beta)

    def compute_td_loss(self, batch_size, frame_idx):

        beta = self.beta_by_frame(frame_idx)

        if len(self.replay_buffer) < batch_size:
            return None

        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(
            batch_size, beta)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))
        weights = Variable(torch.FloatTensor(weights))

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
        loss = loss * weights

        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

        return loss
