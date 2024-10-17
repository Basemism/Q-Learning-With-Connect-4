import random
import pickle

class QLearning:
    
    def __init__(self, player_num):
        self.player = player_num
        self.q = {}
        self.ep = 0.2
        self.lr = 0.8
        self.df = 0.95
        
    def get_q_safe(self, state, action):
        
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = 1.0 #Allow exploration
        return self.q.get((state, action))
    
    
    def make_action(self, state, actions):
        current_state = state
        
        # Exploration
        if random.random() < self.ep:
            return random.choice(actions)

        qList = [self.get_q_safe(current_state, a) for a in actions]
        maxQ = max(qList)

        
        if qList.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best = [i for i in range(len(actions)) if qList[i] == maxQ]
            i = random.choice(best)
        else:
            i = qList.index(maxQ)
        return actions[i]
        
    
    def learn(self, state, action, reward, next_state, next_actions):
        current_q = self.get_q_safe(state, action)
        
        # If the game is over, there's no next action, so the future reward is 0
        if not next_actions:
            future_q = 0
        else:
            future_q = max([self.get_q_safe(next_state, a) for a in next_actions])
        
        # Update the Q-value using the Q-learning formula
        self.q[(state, action)] = current_q + self.lr * (reward + self.df * future_q - current_q)
        
    def save_q_table(self, filename): # Best not to save Q-tables, file sizes get too big,
        with open(filename, 'wb') as f:
            pickle.dump(self.q, f)
        
    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q = pickle.load(f)

        