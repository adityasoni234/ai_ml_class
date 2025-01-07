import gymnasium as gym  # Just change this line, rest remains same
import numpy as np
import warnings

# Setup environment
env = gym.make('CartPole-v1', render_mode='human')

# Set environment at initial state
state = env.reset()

print("State space:", env.observation_space)  # open a ui with game
print("Action space:", env.action_space)

"""
Observation: [-0.10655332 -0.23713252 0.10257257 0.37649268] is the current state of the environment, which could represent:

    Position of the cart
    Velocity of the cart
    Angle of the pole
    Angular velocity of the pole

    Observation: [-0.03149871 -0.5739105  -0.02245608  0.48178494]
"""

for _ in range(1000):
    env.render()  # Render the environment for visualization
    action = env.action_space.sample()  # Take a random action

    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Action: {action}, Reward: {reward}, Observation: {observation}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

    # Check if episode is done (either terminated or truncated)
    if terminated or truncated:
        observation, info = env.reset()  # Reset the environment if the episode is finished

env.close()  # Close the environment when done
