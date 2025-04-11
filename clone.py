import torch
import random
import numpy as np
import pandas as pd
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Load mô hình nếu có file checkpoint
        model_path = "C:\\Users\\votie\\Downloads\\snake-ai-pytorch-main\\model\\model.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            #self.model.eval()
            print("Model loaded successfully!")


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
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    game_100_data = []
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
        if agent.n_games == 99 and not done:  # 99 vì n_games tăng sau khi reset
            # Xác định Danger
            if state_old[0] == 1:
                danger = "no danger"
            elif state_old[1] == 1:
                danger = "danger right"
            elif state_old[2] == 1:
                danger = "danger left"
            else:
                danger = "no danger"

            # Xác định Direction
            direction_mapping = ["left", "right", "up", "down"]
            direction = direction_mapping[np.argmax(state_old[3:7])] if np.any(state_old[3:7]) else "unknown"

            # Xác định Food Direction
            food_x = "left" if state_old[7] else ("right" if state_old[8] else "")
            food_y = "up" if state_old[9] else ("down" if state_old[10] else "")
            food_direction = f"{food_x} {food_y}".strip() if food_x or food_y else "unknown"

            # Xác định Action
            action_mapping = ["left", "straight", "right"]
            action_str = action_mapping[final_move.index(1)] if 1 in final_move else "unknown"
            

            game_100_data.append({
                'state': state_old.tolist(),  
                'action': final_move,
                'Danger': danger,
                'Direction': direction,
                'Food direction': food_direction,
                'Action': action_str
            })


        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Khi kết thúc game 100, lưu dữ liệu vào Excel
            if agent.n_games == 100:
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