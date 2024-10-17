# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:27:17 2024

@author: Basem
"""
from connect_4_env import Connect4
from DQNAgent import DQNAgent
from RandomAgent import RandomAgent
from tqdm import tqdm
import matplotlib.pyplot as plt

env = Connect4()
agent = DQNAgent()
agent.load('./models/connect4_dqn_17000_v5.h5')
agent.epsilon = agent.epsilon_min # Change if loaded
e= 17000 # epoch start
r_agent = RandomAgent()


# Training parameters
n_episodes = 7500
batch_size = 32

# Save Settings
save_rate = 500

# Stats
epoch_list=[]
win_list=[]
wins=0
draws=0

for i in tqdm(range(e, e + n_episodes)):
    state = env.reset()
    done = False
    while not done:
        # DQN agent's turn (player 1)
        valid_actions = env.get_free_cols()
        action = agent.act(state, valid_actions)
        next_state, reward, done = env.make_move(action, 1)
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            # print(f"episode: {i}/{n_episodes}, DQN agent won (Reward {reward}), epsilon: {agent.epsilon:.2}")
            wins+=1
            break
        
        # Random opponent's turn (player 2)
        valid_actions = env.get_free_cols()
        opponent_action = r_agent.make_action(state, valid_actions)
        next_state, reward, done = env.make_move(opponent_action, 2)
        
        agent.remember(state, action, -reward, next_state, done)
        state = next_state

        if done and reward == 1:
            # print(f"episode: {i}/{n_episodes}, Opponent won (Reward {-reward}), epsilon: {agent.epsilon:.2}")
            break
    
    else:
        draws+=1

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # print and save states
    if i % 100 == 99:
        print("Win percentage: {:.2f}%".format(wins))
        print("Draw percentage: {:.2f}%".format(draws))
        win_list.append(wins)
        epoch_list.append(i+1)
        wins, draws = 0, 0
        
    # Save the model every {save_rate} episodes    
    if i % save_rate == save_rate-1:
        agent.save(f"./models/connect4_dqn_{i+1}_v5.h5")
        
print("Training finished.")

# Plotting win percentage against epochs
plt.figure(figsize=(10, 6))
plt.plot(epoch_list, win_list, marker='o', color='b')
plt.xlabel('Epochs')
plt.ylabel('Win Percentage')
plt.title('Win Percentage Over Time')
plt.grid(True)
plt.legend()
plt.savefig('win_percentage_vs_epochs2.png')
plt.show()
