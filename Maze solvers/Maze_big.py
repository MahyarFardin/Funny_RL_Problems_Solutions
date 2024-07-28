import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Maze():
    def __init__(self):
        self.maze = np.array([
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0, 2]
        ], dtype = np.int8)
        self.reward = 1
        self.punish = -1
        self.gamma = 0.5
        self.start = [0, 0]
        self.goal = (9, 9)
        self.current_pos = self.start
        self.n_actions = 4
        self.current_steps = 0

    def reset(self):
        self.current_pos = self.start
        self.current_action = 0
        return self.current_pos

    def step(self, action):
        done = False
        self.current_steps += 1
        new_pos = self.current_pos.copy()

        if action == 0:  # right
            new_pos[1] += 1
        elif action == 1:  # down
            new_pos[0] += 1
        elif action == 2:  # left
            new_pos[1] -= 1
        elif action == 3:  # up
            new_pos[0] -= 1

        if new_pos[0] < 0 or new_pos[0] >= 10 or new_pos[1] < 0 or new_pos[1] >= 10 or self.maze[new_pos[0], new_pos[1]] == 1:
            reward = self.punish
            new_pos = self.current_pos
        elif self.maze[new_pos[0], new_pos[1]] == 2:
            reward = self.reward
            done = True
        else:
            reward = 0
            done = False

        self.current_pos = new_pos

        if tuple(self.current_pos) == self.goal:
            done = True
            reward = self.reward

        return self.current_pos, reward, done
        
    def plot(self):
        maze_copy = self.maze.copy()
        maze_copy[self.current_pos[0], self.current_pos[1]] = -1
        
        sns.heatmap(maze_copy, annot=True, cbar=False)
        plt.show()
    
    def simulate(self, position, action):
        current = self.current_pos
        self.current_pos = position
        new_pos, reward, done = self.step(action)
        self.current_pos = current
        return new_pos, reward, done