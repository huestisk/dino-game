import torch
from dinoEnv import DinoEnv

# Use GPU, if available
USE_CUDA = torch.cuda.is_available()


class DinoAI():

    def __init__(self):
        self.env = DinoEnv()
        self.load_model()
        self.state = self.env.reset()

    def load_model(self):
        try:
            if USE_CUDA:
                self.model = torch.load("training/model.pkl")
            else:
                self.model = torch.load(
                    "training/model.pkl", map_location={'cuda:0': 'cpu'})
        except FileNotFoundError as e:
            print(e)
            return None

    def play(self):
        action = self.model.act(self.state)
        self.state, img, done = self.env.step(action, True)
        return img, done

if __name__ == "__main__":

    ai = DinoAI()
    ai.play()
