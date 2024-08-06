# Mountain Car Descrete Environment for Reinforcement Learning
import os
import gymnasium as gym
import numpy as np

def get_state(observation):
    position, velocity = observation
    position_state = np.digitize(position, position_bins) - 1 
    velocity_state = np.digitize(velocity, velocity_bins) - 1
    return (position_state, velocity_state)

def epsilon_greedy_policy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        return np.argmax(q_table[state])

env = gym.make('MountainCar-v0', render_mode='human')

action_size = env.action_space.n

num_of_bins = 10

position_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_of_bins)
velocity_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num_of_bins)

q_table = np.zeros((num_of_bins, num_of_bins, action_size))
# Load Q-table if it exists
if os.path.exists("q_table_latest.npy"):
    q_table = np.load("q_table_latest.npy")
    print("Loaded Q-table from q_table_latest.npy")


learning_rate = 0.1
discount_rate = 0.99
episodes = 10000

for i in range(episodes):
    observation, info = env.reset()
    state = get_state(observation=observation)
    done = False
    while not done:
        action = epsilon_greedy_policy(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = get_state(next_state)
        q_table[state][action] += learning_rate * (reward + discount_rate * np.max(q_table[next_state]) - q_table[state][action])
        state = next_state
    
    if i % 1000 == 0:
        print(f"Episode {i}")
        env.render()
        np.save(f"q_table_{i}.npy", q_table)

env.close()