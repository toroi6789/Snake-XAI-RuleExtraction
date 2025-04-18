import torch
import random
import numpy as np
import pandas as pd
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.record = 0
        self.total_score = 0
        self.plot_scores = []
        self.plot_mean_score = []

    def save(self, path="C:\\Users\\admin\\Desktop\\snake-ai-pytorch\\model\\model.pth"):
        # Lưu trạng thái của agent
        state = {
            'model_state_dict': self.model.state_dict(),  # Lưu Q-network
            'memory': list(self.memory),  # Lưu replay buffer
            'epsilon': self.epsilon,  # Lưu epsilon
            'gamma': self.gamma,
            'n_games' : self.n_games,# Lưu gamma
            'record' : self.record,
            'total_score' : self.total_score,
            'plot_mean_score' : self.plot_mean_score,
            'plot_scores' : self.plot_scores
        }
        torch.save(state, path)
        print(f"Đã lưu trạng thái agent")

    def load(self, path="C:\\Users\\admin\\Desktop\\snake-ai-pytorch\\model\\model.pth"):
        # Load trạng thái của agent
        state = torch.load(path, weights_only=False)
        self.model.load_state_dict(state['model_state_dict'])
        self.memory = deque(state['memory'], maxlen=100000)
        self.epsilon = state['epsilon']
        self.gamma = state['gamma']
        self.n_games = state['n_games']
        self.record = state['record']
        self.total_score = state['total_score']
        self.plot_scores = state['plot_scores']
        self.plot_mean_score = state['plot_mean_score']
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Khởi tạo lại trainer với gamma đã load
        print(f"Đã load trạng thái agent từ {path}, epsilon: {self.epsilon}, n_games: {self.n_games}, total_scores: {self.total_score}, plot_scores: {self.plot_scores}, plot-mean_score: {self.plot_mean_score}, average: {self.total_score/ self.n_games}")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
   
    agent = Agent()
    #Load nếu có file save
    if os.path.exists("C:\\Users\\admin\\Desktop\\snake-ai-pytorch\\model\\model.pth"):
        agent.load("C:\\Users\\admin\\Desktop\\snake-ai-pytorch\\model\\model.pth")
    game = SnakeGameAI()
    game_100_data = []
    gamesPlayed = agent.n_games
    record = agent.record
    total_score = agent.total_score
    plot_scores = agent.plot_scores
    plot_mean_scores = agent.plot_mean_score
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # Nếu là game thứ 100, lưu state và action
        if agent.n_games == gamesPlayed + 99 and not done:  # 99 vì n_games tăng sau khi reset
            game_100_data.append({
                'state': state_old.tolist(),  # Lưu state cũ
                'action': final_move          # Lưu action
            })

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                
                agent.record = record
                agent.total_score = total_score
                agent.plot_scores = plot_scores
                agent.plot_mean_score = plot_mean_scores
                agent.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Khi kết thúc game thu 100, lưu dữ liệu vào Excel
            if agent.n_games - gamesPlayed == 100:
                try:
                    df = pd.DataFrame(game_100_data)
                    df.to_excel('game_100_data.xlsx', index=False, engine='openpyxl')
                    print("Đã lưu dữ liệu game 100 vào 'game_100_data.xlsx'")
                except Exception as e:
                    print(f"Lỗi khi lưu file Excel: {e}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()