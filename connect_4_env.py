# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:10:22 2024

@author: Basem
"""

import numpy as np
from colorama import Fore, Style

INVALID = -10
DRAW = 0.5
WIN = 1
PROGRESS = 0

class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board_state =  np.zeros((self.rows, self.cols))
        self.isOver = False
        
    def reset(self):
        self.__init__()
        return np.reshape(self.board_state, [1,6,7])

    def get_free_cols(self):
        return [ c for c in range(self.cols) if self.board_state[self.rows-1, c] == 0]
    
    def check_game_status(self, player, r, c):
        directions = [
        (0, 1),  # right
        (1, 0),  # down
        (1, 1),  # down-right
        (1, -1)  # down-left
        ]
        
        for dr, dc in directions:
            count = 1
            
            #Check Positive
            for i in range(1, 4):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board_state[nr][nc] == player:
                    count += 1
                else:
                    break
                
            #Check Negative
            for i in range(1, 4):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board_state[nr][nc] == player:
                    count += 1
                else:
                    break
 
            if count >= 4:
                return True
        
        return False
    
    def make_move(self, col, player): # player = 1 or 2
        if col not in self.get_free_cols():
            return self.board_state.copy(), INVALID, self.isOver
        
        for row in range(0, len(self.board_state)):
            if self.board_state[row, col] == 0:
                self.board_state[row, col] = player
                break 
        
        if self.check_game_status(player, row, col):
            self.isOver = True
            reward = WIN
            
        elif not self.get_free_cols():
            self.isOver = True
            reward = DRAW
            
        else:
            reward = PROGRESS
            
        return np.reshape(self.board_state.copy(), [1, 6, 7]), reward, self.isOver
            
    
    def render(self):
        string = []
        red, blue = Fore.RED, Fore.BLUE
        
        for row in reversed(self.board_state):
            row_string = []
            for x in row:
                if x == 0:  
                    row_string.append(Style.RESET_ALL + "⬤ ")
                elif x == 1:
                    row_string.append(red + "⬤ ")
                else:
                    row_string.append(blue + "⬤ ")
            string.append(''.join(row_string))
        string.append(Style.RESET_ALL)
            
        print('\n'.join(string))
            

if __name__ == "__main__":
    colour=['red','blue']
    env = Connect4()
    done = False
    while not done:
        for player in [1,2]:
            env.render()
            action = int(input(f"Player {player}'s turn. You are {colour[player-1]}. Choose a column (0-6): "))
            _, r, done = env.make_move(action, player)
            
            if done:
                env.render()
                if r == WIN:
                    print(f"Player {player} wins!")
                else:
                    print("Game Over by Draw")
                break
