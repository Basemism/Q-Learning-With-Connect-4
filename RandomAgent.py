import random

class RandomAgent:

    def make_action(self, state, actions):
        return random.choice(actions)
