import gym
from gym import spaces

import pygame.surfarray as s
from dinoGame import DinoGame

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import cv2
import numpy as np

N_CHANNELS, WIDTH, HEIGHT = 4, 150, 600

class DinoEnv(gym.Env):

    def __init__(self):
        super(DinoEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # None, Jump, Crouch, Standup
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(N_CHANNELS, WIDTH, HEIGHT), dtype=np.uint8)

        self.game = DinoGame(fps=30, render=False)
        self._state = np.zeros((N_CHANNELS, WIDTH, HEIGHT))
        
    def reset(self, color=True):
        self.game.reset()
        self.game.start_running()
        self.game._update_state()
        self.update_state()
        return self._state

    def step(self, action):
        info = dict()
        done = self.game._ai_loop(action)
        reward = 0.01 if not done else 0
        img = self.update_state()
        return self._state, reward, done, info

    def update_state(self):
        # Render the current state
        self.game._draw()
        # Get image in pygame and convert
        img = s.array3d(self.game._screen).swapaxes(0, 1)
        # cv2.imshow('', img)
        # cv2.waitKey(1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=0)
        self._state = np.concatenate((img, self._state[1:, :, :]))

if __name__ == "__main__":

    env = DinoEnv()
    state = env.reset()

    for i in range(1000):
        if i % 50 == 0:
            env.step(1)
        else:
            env.step(0)
        

    print(0)