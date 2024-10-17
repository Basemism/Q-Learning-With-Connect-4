# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:45:22 2024

@author: Basem
"""

import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    def __init__(self):
        self.state_size = (6,7)
        self.action_size = 7
        self.memory = deque(maxlen=10000) #buffer size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # lower bound
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    # Build a 3 layer nueral network with 128 units in the hidden layer
    # Neural network uses mean squared error loss function and Adam optimizer
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.state_size),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    # Stores an event to agents memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Selects an action, by exploration or exploitation
    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state = np.reshape(state, (1, 6, 7))  # Ensure shape is (1, 6, 7)
        q_values = self.model.predict(state, verbose=0)
        return valid_actions[np.argmax([q_values[0][i] for i in valid_actions])]

    # Training method, samples a random batch of events (in memory), updates Q-values (via Bellman's Equation), and decays exploration rate. 
    # If event leads to finishing state, the target is the reward.
    # Other wise the target is reward + discounted maximum future reward.
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([item[0] for item in minibatch]).reshape(batch_size, 6, 7)
        next_states = np.array([item[3] for item in minibatch]).reshape(batch_size, 6, 7)
        
        state_predictions = self.model.predict(states, verbose=0)
        next_state_predictions = self.model.predict(next_states, verbose=0)
        targets = state_predictions.copy()
        

        for i, (state, action, reward, _, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                max_future_q = np.amax(next_state_predictions[i])
                targets[i][action] = reward + self.gamma * max_future_q
        
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

