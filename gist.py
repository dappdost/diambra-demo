import gym
import diambra.arena
from diambra.arena.stable_baselines3.utils import make_env

# Create the environment
env = make_env(game="sfiii3n", characters=["Ryu"], roles=["P1"], frame_shape=(84, 84))
env = gym.wrappers.RecordVideo(env, "videos/", step_trigger=lambda x: x % 10000 == 0)

# Initialize environment
observation = env.reset()
done = False

# Run random agent for 5000 steps
for _ in range(5000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

# Close environment
env.close()
