# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:01:20 2024

@author: Basem
"""
from connect_4_env import Connect4
from DQNAgent import DQNAgent
from RandomAgent import RandomAgent
import time


env = Connect4()
agent = DQNAgent()
agent.load('./models/connect4_dqn_19000_v5.h5')
agent.epsilon = agent.epsilon_min
r_agent = RandomAgent()

while True:
    try:
        state = env.reset()
        done = False
        while not done:
            # DQN agent's turn (player 1)
            valid_actions = env.get_free_cols()
            action = agent.act(state, valid_actions)
            next_state, reward, done = env.make_move(action, 1)
            
            if done:
                print("You Lose!")
                env.render()
                break
            
            
            
            # Random opponent's turn (player 2)
            action = r_agent.make_action(next_state, env.get_free_cols())
            
            # env.render()
            # action = int(input(f"Your Turn. You are Blue {env.get_free_cols()}"))
            next_state, reward, done = env.make_move(action, 2)
            state = next_state

            if done:
                print('You Won!')
                env.render()
                break
            
            env.render()
            time.sleep(1)
            
        else: 
            print("It's a draw!")
    except KeyboardInterrupt():
        exit()

    