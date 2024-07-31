import numpy as np
import random
import copy
from scenarios.base_env import BaseScenario


class PathPlanningEnv(BaseScenario):
    def __init__(self, n_agent, map_size=20, obstacle_density=0.2, detection_radius=0.2, start_pos=None, end_pos=None):
        super().__init__()
        self.n_agent = n_agent
        self.map_size = map_size
        self.obstacle_density = obstacle_density
        self.detection_radius = detection_radius  # 检测半径
        self.map = self.generate_map()
        self.start_pos = start_pos if start_pos else self.generate_positions()
        self.end_pos = end_pos if end_pos else self.generate_positions()
        self.reached_end = [False] * self.n_agent  # 标记目标点是否已完成
        self.shared_obstacles = np.zeros_like(self.map)  # 共享的障碍物信息
        self.shared_targets = np.zeros((self.n_agent, 2))  # 共享的目标信息
        self.reset()

    def generate_map(self):
        # 生成一个map，其中1代表障碍物，0代表空地
        map = np.zeros((self.map_size, self.map_size))
        num_obstacles = int(self.map_size * self.map_size * self.obstacle_density)
        obstacles = random.sample(list(np.ndindex(self.map_size, self.map_size)), num_obstacles)
        for obs in obstacles:
            map[obs] = 1
        return map

    def generate_positions(self):
        positions = []
        while len(positions) < self.n_agent:
            pos = np.random.randint(0, self.map_size, size=2)
            if self.map[pos[0], pos[1]] == 0 and pos.tolist() not in positions:
                positions.append(pos.tolist())
        return np.array(positions) / self.map_size  # 将位置归一化为[0, 1]之间

    def reset(self):
        self.agent_pos = copy.deepcopy(self.start_pos)
        self.path_history = [[] for _ in range(self.n_agent)]
        self.done = [False] * self.n_agent
        self.end_pos = self.generate_positions()  # Generate new target position
        self.reached_end = [False] * self.n_agent  # 重置目标完成标记
        self.shared_obstacles = np.zeros_like(self.map)  # 重置共享的障碍物信息
        self.shared_targets = np.zeros((self.n_agent, 2))  # 重置共享的目标信息
        return self.get_obs(), self.get_adj()

    def get_obs(self):
        obs = []
        for i in range(self.n_agent):
            current_pos = self.agent_pos[i]
            goal_pos = self.shared_targets[i] if self.shared_targets[i].any() else self.end_pos[i]
            obs.append(np.concatenate([current_pos, goal_pos]))
        return np.array(obs)

    def get_adj(self):
        adj = np.zeros((self.n_agent, self.n_agent))
        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i != j:
                    adj[i, j] = 1 / (np.linalg.norm(self.agent_pos[i] - self.agent_pos[j]) + 1e-5)
        return adj

    def detect_obstacles_and_targets(self):
        detected_obstacles = np.zeros_like(self.map)
        detected_targets = np.zeros((self.n_agent, 2))
        for i in range(self.n_agent):
            pos = self.agent_pos[i]
            grid_pos = (pos * self.map_size).astype(int)
            for x in range(max(0, grid_pos[0] - int(self.detection_radius * self.map_size)),
                           min(self.map_size, grid_pos[0] + int(self.detection_radius * self.map_size) + 1)):
                for y in range(max(0, grid_pos[1] - int(self.detection_radius * self.map_size)),
                               min(self.map_size, grid_pos[1] + int(self.detection_radius * self.map_size) + 1)):
                    if np.linalg.norm(pos - np.array([x, y]) / self.map_size) <= self.detection_radius:
                        detected_obstacles[x, y] = self.map[x, y]
                        if np.linalg.norm(self.end_pos[i] - np.array([x, y]) / self.map_size) <= self.detection_radius:
                            detected_targets[i] = self.end_pos[i]
        return detected_obstacles, detected_targets

    def update_shared_info(self, detected_obstacles, detected_targets):
        self.shared_obstacles = np.maximum(self.shared_obstacles, detected_obstacles)
        for i in range(self.n_agent):
            if detected_targets[i].any():
                self.shared_targets[i] = detected_targets[i]

    def step(self, actions):
        actions = np.clip(actions, -0.005, 0.005)  # Shorten length
        rewards = np.zeros(self.n_agent)
        detected_obstacles, detected_targets = self.detect_obstacles_and_targets()
        self.update_shared_info(detected_obstacles, detected_targets)
        for i in range(self.n_agent):
            if not self.done[i]:
                new_pos = self.agent_pos[i] + actions[i]
                new_pos = np.clip(new_pos, 0, 1)  # Ensure new target is in the field
                grid_pos = (new_pos * self.map_size).astype(int)
                if self.map[grid_pos[0], grid_pos[1]] == 0:  # Ensure new target is not obstacle
                    self.agent_pos[i] = new_pos
                    self.path_history[i].append(self.agent_pos[i].copy())

                # Check moved towarrd destination or away
                if np.linalg.norm(self.agent_pos[i] - self.end_pos[i]) < 1.0 / self.map_size:
                    self.done[i] = True
                    self.reached_end[i] = True  # Label destination arrived change the status
                    rewards[i] += 100  # BIG award
                    self.end_pos[i] = self.generate_positions()[i]  # Generate a new target objective
                else:
                    rewards[i]-=0.01#cost

                # Check objecti in range for obstacle and tartgets
                if np.linalg.norm(self.agent_pos[i] - self.shared_targets[i]) < self.detection_radius:
                    rewards[i] += 10  # ADD point when in range
                    direction_to_target = self.shared_targets[i] - self.agent_pos[i]
                    actions[i] = direction_to_target / np.linalg.norm(direction_to_target) * 0.005  # move toward objective after sharing

                if self.shared_obstacles[grid_pos[0], grid_pos[1]] == 1:
                    rewards[i] -= 10  # reduce when obstacles met
                    direction_away_from_obstacle = self.agent_pos[i] - np.array(grid_pos) / self.map_size
                    actions[i] = direction_away_from_obstacle / np.linalg.norm(
                        direction_away_from_obstacle) * 0.005  # move away from obstacle

        obs = self.get_obs()
        adj = self.get_adj()
        return rewards, obs, adj, self.done

    def render(self):
        import matplotlib.pyplot as plt
        plt.clf()
        for i in range(self.n_agent):
            path = np.array(self.path_history[i])
            if len(path) > 0:
                plt.plot(path[:, 0] * self.map_size, path[:, 1] * self.map_size, label=f'Agent {i}')
            # PLOT
            circle = plt.Circle((self.agent_pos[i, 0] * self.map_size, self.agent_pos[i, 1] * self.map_size),
                                self.detection_radius * self.map_size, color='blue', fill=False, linestyle='dashed')
            plt.gca().add_artist(circle)
        plt.scatter(self.start_pos[:, 0] * self.map_size, self.start_pos[:, 1] * self.map_size, c='red', label='Start')
        plt.scatter(self.end_pos[:, 0] * self.map_size, self.end_pos[:, 1] * self.map_size, c='green', label='End')
        for i, reached in enumerate(self.reached_end):
            if reached:
                plt.scatter(self.end_pos[i, 0] * self.map_size, self.end_pos[i, 1] * self.map_size, c='yellow',
                            label='Reached')
        obs_indices = np.argwhere(self.map == 1)
        plt.scatter(obs_indices[:, 1], obs_indices[:, 0], c='black', label='Obstacle')
        plt.legend()
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.pause(0.01)
