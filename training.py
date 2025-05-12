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
    
    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings, render_mode="human")
    print("Activated {} environment(s)".format(num_envs))

    # Create model
    if cfg["ppo_settings"].get("model_checkpoint", "0") != "0":
        model = PPO.load(cfg["ppo_settings"]["model_checkpoint"], env=env)
        print(f"Loaded model from {cfg['ppo_settings']['model_checkpoint']}")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            batch_size=cfg["ppo_settings"]["batch_size"],
            n_steps=cfg["ppo_settings"]["n_steps"],
            learning_rate=cfg["ppo_settings"]["learning_rate"][0],
            gamma=cfg["ppo_settings"]["gamma"],
            ent_coef=0.01,
            clip_range=cfg["ppo_settings"]["clip_range"][0],
            n_epochs=cfg["ppo_settings"]["n_epochs"],
            gae_lambda=0.95,
            max_grad_norm=0.5,
            vf_coef=0.5,
            device="auto",
            policy_kwargs=cfg["policy_kwargs"]
        )

    # Create output directory
    output_dir = os.path.join(
        cfg["folders"]["parent_dir"],
        cfg["settings"]["game_id"],
        cfg["folders"]["model_name"],
        "model"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Define callback for model saving
    autosave_freq = cfg["ppo_settings"].get("autosave_freq", 0)
    if autosave_freq > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback
        checkpoint_callback = CheckpointCallback(
            save_freq=autosave_freq,
            save_path=output_dir,
            name_prefix="checkpoint"
        )
        callbacks = [checkpoint_callback]
    else:
        callbacks = []

    # Train model
    model.learn(
        total_timesteps=cfg["ppo_settings"]["time_steps"],
        callback=callbacks
    )

    # Save final model
    model.save(os.path.join(output_dir, "model"))

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
