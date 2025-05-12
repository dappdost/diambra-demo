import os
import sys
import yaml
import argparse
import numpy as np
from stable_baselines3 import PPO
import gym
import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena import load_settings_flat_dict, SpaceTypes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgFile", type=str, required=True, help="Configuration file")
    parser.add_argument("--modelFile", type=str, required=True, help="Model file")
    parser.add_argument("--numEpisodes", type=int, default=10, help="Number of episodes")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    with open(args.cfgFile, 'r') as f:
        cfg = yaml.safe_load(f)

    # Settings
    cfg["settings"]["action_space"] = SpaceTypes.DISCRETE if cfg["settings"]["action_space"] == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, cfg["settings"])

    # Wrappers Settings
    wrappers_settings = load_settings_flat_dict(WrappersSettings, cfg["wrappers_settings"])
    
    # Create environment with recording and rendering
    wrappers_settings.normalize_reward = False  # For evaluation, use raw rewards
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, render_mode="human")
    print("Activated {} environment(s)".format(num_envs))

    # Load model
    model = PPO.load(args.modelFile, env=env)

    # Evaluation loop
    episode_rewards = []
    for episode in range(args.numEpisodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Print progress
            if reward != 0:
                print(f"Step {step}, Reward: {reward}, Total: {total_reward}")

        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}/{args.numEpisodes}, Total Reward: {total_reward}")

    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Minimum Reward: {np.min(episode_rewards):.2f}")
    print(f"Maximum Reward: {np.max(episode_rewards):.2f}")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
