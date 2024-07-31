import argparse
import yaml
import numpy as np
from scenarios.path_planning import PathPlanningEnv
from agents.dqn import DQNAgent
#tag

def main(args):
    with open(f'configs/{args.scenario}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = PathPlanningEnv(n_agent=config['n_agent'], map_size=config['map_size'],
                          obstacle_density=config['obstacle_density'], detection_radius=config['detection_radius'])
    agent = DQNAgent(obs_dim=4, action_dim=2,
                     gamma=config['gamma'], epsilon=config['epsilon'],
                     epsilon_min=config['epsilon_min'], epsilon_decay=config['epsilon_decay'],
                     learning_rate=config['learning_rate'])


    for episode in range(config['episodes']):
        obs, adj = env.reset()
        done = [False] * config['n_agent']
        total_reward = 0
        while not all(done):
            actions = np.array([agent.act(o) for o in obs])
            rewards, next_obs, adj, done = env.step(actions)
            for i in range(config['n_agent']):
                agent.remember(obs[i], actions[i], rewards[i], next_obs[i], done[i])
            obs = next_obs
            total_reward += np.sum(rewards)
            if len(agent.memory) > config['batch_size']:
                agent.replay(config['batch_size'])
            env.render()
        agent.update_target_model()
        print(f"Episode {episode + 1}/{config['episodes']}, Total Reward: {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default="path_planning")
    args = parser.parse_args()
    main(args)
