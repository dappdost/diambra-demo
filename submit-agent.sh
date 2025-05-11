#!/bin/bash

# Text formatting
bold=$(tput bold)
normal=$(tput sgr0)
green=$(tput setaf 2)
red=$(tput setaf 1)
yellow=$(tput setaf 3)

echo "${bold}==================================================="
echo "Diambra Street Fighter III AI Agent - WSL Setup"
echo "===================================================${normal}"

# Get current directory
PROJECT_DIR=$(pwd)
echo "Using project directory: $PROJECT_DIR"

# Step 1: Install system dependencies in WSL
echo
echo "${bold}STEP 1: Installing system dependencies in WSL${normal}"
echo "==================================================="

# Ensure apt repositories are up to date
echo "Updating apt repositories..."
sudo apt update

# Install required system packages
echo "Installing required system packages..."#!/bin/bash

# Text formatting
bold=$(tput bold)
normal=$(tput sgr0)
green=$(tput setaf 2)
red=$(tput setaf 1)
yellow=$(tput setaf 3)

echo "${bold}==================================================="
echo "Diambra Street Fighter III AI Agent - WSL Setup"
echo "===================================================${normal}"

# Get current directory
PROJECT_DIR=$(pwd)
echo "Using project directory: $PROJECT_DIR"

# Step 1: Install Python 3.9 and system dependencies in WSL
echo
echo "${bold}STEP 1: Installing Python 3.9 and system dependencies in WSL${normal}"
echo "==================================================="

# Ensure apt repositories are up to date
echo "Updating apt repositories..."
sudo apt update

# Install required system packages
echo "Installing required system packages..."
sudo apt install -y software-properties-common build-essential curl wget

# Add deadsnakes PPA for Python 3.9
echo "Adding PPA for Python 3.9..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.9 and development tools
echo "Installing Python 3.9 and development tools..."
sudo apt install -y python3.9 python3.9-venv python3.9-dev

# Verify Python 3.9 installation
if ! command -v python3.9 &> /dev/null; then
    echo "${red}${bold}ERROR: Python 3.9 installation failed.${normal}"
    exit 1
else
    echo "${green}Python 3.9 installed successfully: $(python3.9 --version)${normal}"
fi

# Install pip for Python 3.9
echo "Installing pip for Python 3.9..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py
rm get-pip.py

# Verify pip installation
if ! python3.9 -m pip --version &> /dev/null; then
    echo "${red}${bold}ERROR: pip installation for Python 3.9 failed.${normal}"
    exit 1
else
    echo "${green}pip installed successfully: $(python3.9 -m pip --version)${normal}"
fi

# Check if Docker is available in WSL
echo "Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    echo "${yellow}Docker is not installed in WSL.${normal}"
    echo "Docker should be available through Docker Desktop for Windows."
    echo "Make sure Docker Desktop is running with WSL integration enabled."
    
    # Check if docker can be accessed
    docker info &> /dev/null
    if [ $? -ne 0 ]; then
        echo "${red}${bold}ERROR: Cannot access Docker.${normal}"
        echo "Please ensure Docker Desktop is running with WSL integration enabled."
        echo "In Docker Desktop settings, go to Resources > WSL Integration and enable it for this WSL distro."
        exit 1
    else
        echo "${green}Docker is accessible through Windows Docker Desktop.${normal}"
    fi
else
    echo "${green}Docker is installed in WSL.${normal}"
fi

# Step 2: Create and activate virtual environment with Python 3.9
echo
echo "${bold}STEP 2: Creating Python 3.9 virtual environment${normal}"
echo "==================================================="

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment with Python 3.9..."
    python3.9 -m venv venv
    if [ $? -ne 0 ]; then
        echo "${red}${bold}ERROR: Failed to create virtual environment.${normal}"
        echo "Trying to create with system packages..."
        python3.9 -m venv venv --system-site-packages
        if [ $? -ne 0 ]; then
            echo "${red}${bold}ERROR: Virtual environment creation failed. Cannot continue.${normal}"
            exit 1
        fi
    fi
else
    echo "Virtual environment already exists."
    
    # Check if it was created with Python 3.9
    if [ -f "$PROJECT_DIR/venv/pyvenv.cfg" ]; then
        VENV_PYTHON_VERSION=$(grep "version" "$PROJECT_DIR/venv/pyvenv.cfg" | cut -d "=" -f 2 | tr -d " ")
        if [[ ! "$VENV_PYTHON_VERSION" =~ ^3\.9\. ]]; then
            echo "${yellow}${bold}WARNING: Existing virtual environment uses Python $VENV_PYTHON_VERSION, not 3.9.${normal}"
            echo "Recreating virtual environment with Python 3.9..."
            rm -rf "$PROJECT_DIR/venv"
            python3.9 -m venv venv
            if [ $? -ne 0 ]; then
                echo "${red}${bold}ERROR: Failed to recreate virtual environment with Python 3.9.${normal}"
                exit 1
            fi
        fi
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "${red}${bold}ERROR: Failed to activate virtual environment.${normal}"
    exit 1
fi

# Verify virtual environment is active and using Python 3.9
if [ -z "$VIRTUAL_ENV" ]; then
    echo "${red}${bold}ERROR: Virtual environment is not active.${normal}"
    exit 1
else
    echo "${green}Virtual environment is active: $VIRTUAL_ENV${normal}"
fi

# Check Python version in virtual environment
VENV_PYTHON_VERSION=$(python --version 2>&1)
if [[ ! "$VENV_PYTHON_VERSION" =~ Python\ 3\.9\. ]]; then
    echo "${red}${bold}ERROR: Virtual environment is not using Python 3.9.${normal}"
    echo "Current version: $VENV_PYTHON_VERSION"
    exit 1
else
    echo "${green}Virtual environment is using $VENV_PYTHON_VERSION${normal}"
fi

# Install Python dependencies
echo "Installing required Python packages..."
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install numpy==1.23
python -m pip install diambra
python -m pip install diambra-arena
python -m pip install "diambra-arena[stable-baselines3]"

# Verify Diambra CLI is installed
if ! command -v diambra &> /dev/null; then
    echo "${red}${bold}ERROR: Diambra CLI is not in PATH even after installation.${normal}"
    echo "Trying to install Diambra CLI directly..."
    
    # Try to install directly into venv bin directory
    python -m pip install --upgrade diambra
    
    # Add the venv bin directory to PATH for this script
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    
    # Check again
    if ! command -v diambra &> /dev/null; then
        echo "${red}${bold}ERROR: Diambra CLI installation failed. Cannot continue.${normal}"
        
        # Try one more approach - define the full path to diambra
        DIAMBRA_CLI="$VIRTUAL_ENV/bin/diambra"
        if [ ! -f "$DIAMBRA_CLI" ]; then
            echo "${red}${bold}ERROR: Diambra CLI not found at expected location ($DIAMBRA_CLI).${normal}"
            exit 1
        else
            echo "${yellow}Will use full path to Diambra CLI: $DIAMBRA_CLI${normal}"
        fi
    fi
fi

echo "${green}${bold}Diambra CLI is installed: $(diambra --version 2>&1 || echo 'version info not available')${normal}"
echo "${green}${bold}Dependencies installed successfully!${normal}"

# Step 3: Check for ROM file
echo
echo "${bold}STEP 3: Checking ROM file${normal}"
echo "==================================================="

# Create directory if it doesn't exist
mkdir -p "$PROJECT_DIR/roms"

# Check if ROM file exists
if [ ! -f "$PROJECT_DIR/roms/sfiii3n.zip" ]; then
    echo "${yellow}${bold}ROM file not found: $PROJECT_DIR/roms/sfiii3n.zip${normal}"
    echo "Please copy the Street Fighter III ROM (sfiii3n.zip) to the roms folder."
    echo "You may need to copy it from Windows to WSL."
    echo "In Windows Explorer, you can access the WSL file system by typing: \\\\wsl$"
    echo
    read -p "Press Enter once you've added the ROM file..."
    
    # Check again after user input
    if [ ! -f "$PROJECT_DIR/roms/sfiii3n.zip" ]; then
        echo "${red}${bold}ROM file still not found. Cannot continue.${normal}"
        exit 1
    fi
else
    echo "${green}ROM file found!${normal}"
fi

# Step 4: Create configuration file
echo
echo "${bold}STEP 4: Creating configuration file${normal}"
echo "==================================================="

# Create directory if it doesn't exist
mkdir -p "$PROJECT_DIR/cfg_files/sfiii3n"

# Create configuration file if it doesn't exist
if [ ! -f "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml" ]; then
    echo "Creating configuration file..."
    cat > "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml" << EOF
folders:
  parent_dir: "./results/"
  model_name: "sr6_128x4_das_nc"

settings:
  game_id: "sfiii3n"
  step_ratio: 6
  frame_shape: [128, 128, 0]
  continue_game: 0.0
  action_space: "discrete"
  characters: "Ryu"
  difficulty: 4
  outfits: 1

wrappers_settings:
  normalize_reward: true
  no_attack_buttons_combinations: true
  stack_frames: 4
  dilation: 1
  add_last_action: true
  stack_actions: 12
  scale: true
  exclude_image_scaling: true
  role_relative: true
  flatten: true
  filter_keys: ["action", "own_health", "opp_health", "own_side", "opp_side", "opp_character", "stage", "timer"]

policy_kwargs:
  net_arch: [64, 64]

ppo_settings:
  gamma: 0.94
  model_checkpoint: "0"
  learning_rate: [2.5e-4, 2.5e-6]
  clip_range: [0.15, 0.025]
  batch_size: 256
  n_epochs: 4
  n_steps: 128
  autosave_freq: 512
  time_steps: 1024
EOF
    echo "${green}Configuration file created!${normal}"
else
    echo "Configuration file already exists."
fi

# Step 5: Check if model exists or needs training
echo
echo "${bold}STEP 5: Checking for trained model${normal}"
echo "==================================================="

if [ -f "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip" ]; then
    echo "${green}Found existing trained model!${normal}"
    echo "Location: $PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip"
    
    read -p "Do you want to use this existing model for submission? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "You've chosen to train a new model."
        TRAIN_NEW=true
    else
        TRAIN_NEW=false
    fi
else
    echo "${yellow}No trained model found. Need to train a new model.${normal}"
    TRAIN_NEW=true
fi

# Step 6: Train model if needed
if [ "$TRAIN_NEW" = true ]; then
    echo
    echo "${bold}STEP 6: Training a new model${normal}"
    echo "==================================================="
    
    # Create training script
    echo "Creating training script..."
    cat > "$PROJECT_DIR/training.py" << 'EOF'
import os
import sys
import yaml
import argparse
import numpy as np
from stable_baselines3 import PPO
import gym
import diambra.arena
from diambra.arena.stable_baselines3.wrappers import SB3Wrapper

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

    # Create environment
    env = SB3Wrapper(
        game=cfg["settings"]["game_id"],
        characters=cfg["settings"]["characters"],
        roles=["P1"],
        frame_shape=cfg["settings"]["frame_shape"],
        step_ratio=cfg["settings"]["step_ratio"],
        difficulty=cfg["settings"]["difficulty"]
    )

    # Create model
    if cfg["ppo_settings"].get("model_checkpoint", "0") != "0":
        model = PPO.load(cfg["ppo_settings"]["model_checkpoint"], env=env)
        print(f"Loaded model from {cfg['ppo_settings']['model_checkpoint']}")
    else:
        model = PPO(
            "CnnPolicy",
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
            device="auto"
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
EOF
    
    # Ask for parallel environments
    echo "How many parallel environments do you want to use for training?"
    echo "Higher numbers = faster training but more system resources"
    echo "Recommended: 1-4 for most systems, up to 8 for high-end systems"
    read -p "Enter number (1-8): " parallelEnvs
    
    # Validate input
    if ! [[ "$parallelEnvs" =~ ^[0-9]+$ ]]; then
        parallelEnvs=1
    fi
    
    if [ "$parallelEnvs" -lt 1 ]; then
        parallelEnvs=1
    fi
    
    if [ "$parallelEnvs" -gt 8 ]; then
        parallelEnvs=8
    fi
    
    echo "Starting training with $parallelEnvs parallel environments..."
    echo "This will take some time. Progress will be displayed..."
    
    # Ensure diambra is in the PATH
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    
    # Determine how to call diambra
    if command -v diambra &> /dev/null; then
        DIAMBRA_CLI="diambra"
    else
        DIAMBRA_CLI="$VIRTUAL_ENV/bin/diambra"
    fi
    
    if [ "$parallelEnvs" -le 1 ]; then
        $DIAMBRA_CLI run -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
    else
        $DIAMBRA_CLI run -s "$parallelEnvs" -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
    fi
    
    # Check if training was successful
    if [ ! -f "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip" ]; then
        echo "${red}${bold}ERROR: Training failed. Model file not created.${normal}"
        exit 1
    else
        echo "${green}${bold}Training completed successfully!${normal}"
    fi
else
    echo
    echo "${bold}STEP 6: Using existing model (training skipped)${normal}"
    echo "==================================================="
fi

# Step 7: Submit agent
echo
echo "${bold}STEP 7: Submitting agent to Diambra${normal}"
echo "==================================================="

# Ensure diambra is in the PATH
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Check Diambra CLI again
if command -v diambra &> /dev/null; then
    DIAMBRA_CLI="diambra"
else
    DIAMBRA_CLI="$VIRTUAL_ENV/bin/diambra"
    if [ ! -f "$DIAMBRA_CLI" ]; then
        echo "${red}${bold}ERROR: Diambra CLI not found at expected location. Cannot continue.${normal}"
        exit 1
    fi
fi

# Check Diambra login
echo "Checking Diambra login..."
$DIAMBRA_CLI user info &> /dev/null
if [ $? -ne 0 ]; then
    echo "Logging in to Diambra..."
    $DIAMBRA_CLI user login
fi

# Create submission directory
echo "Preparing submission..."
mkdir -p "$PROJECT_DIR/output/models"

# Copy model file
echo "Copying model file to submission directory..."
cp "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip" "$PROJECT_DIR/output/models/1022.zip"

# Change to submission directory
cd "$PROJECT_DIR/output/models"

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << EOF
stable-baselines3
torch
numpy==1.23
EOF

# Initialize agent
echo "Initializing agent..."
$DIAMBRA_CLI agent init .

# Create unique version
VERSION="v$(date +%Y%m%d%H%M)"
echo "Using version tag: $VERSION"

# Submit agent
echo "Submitting agent to Diambra..."
$DIAMBRA_CLI agent submit --submission.difficulty hard --version $VERSION .

# Return to project directory
cd "$PROJECT_DIR"

echo
echo "${green}${bold}Diambra agent setup and submission complete!${normal}"
echo "You can view your submission status on the Diambra website."
sudo apt install -y python3-pip python3-venv python3-dev build-essential

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "${red}${bold}ERROR: Python3 is not installed and could not be installed automatically.${normal}"
    exit 1
else
    echo "${green}Python is available: $(python3 --version)${normal}"
fi

# Check if Docker is available in WSL
echo "Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    echo "${yellow}Docker is not installed in WSL.${normal}"
    echo "Docker should be available through Docker Desktop for Windows."
    echo "Make sure Docker Desktop is running with WSL integration enabled."
    
    # Check if docker can be accessed
    docker info &> /dev/null
    if [ $? -ne 0 ]; then
        echo "${red}${bold}ERROR: Cannot access Docker.${normal}"
        echo "Please ensure Docker Desktop is running with WSL integration enabled."
        echo "In Docker Desktop settings, go to Resources > WSL Integration and enable it for this WSL distro."
        exit 1
    else
        echo "${green}Docker is accessible through Windows Docker Desktop.${normal}"
    fi
else
    echo "${green}Docker is installed in WSL.${normal}"
fi

# Step 2: Create and activate virtual environment
echo
echo "${bold}STEP 2: Creating Python virtual environment${normal}"
echo "==================================================="

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "${red}${bold}ERROR: Failed to create virtual environment.${normal}"
        echo "Trying to create with system packages..."
        python3 -m venv venv --system-site-packages
        if [ $? -ne 0 ]; then
            echo "${red}${bold}ERROR: Virtual environment creation failed. Cannot continue.${normal}"
            exit 1
        fi
    fi
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "${red}${bold}ERROR: Failed to activate virtual environment.${normal}"
    exit 1
fi

# Verify virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "${red}${bold}ERROR: Virtual environment is not active.${normal}"
    exit 1
else
    echo "${green}Virtual environment is active: $VIRTUAL_ENV${normal}"
fi

# Install Python dependencies
echo "Installing required Python packages..."
pip install --upgrade pip
pip install numpy==1.23
pip install wheel
pip install diambra
pip install diambra-arena
pip install "diambra-arena[stable-baselines3]"

# Verify Diambra CLI is installed
if ! command -v diambra &> /dev/null; then
    echo "${red}${bold}ERROR: Diambra CLI is not in PATH even after installation.${normal}"
    echo "Trying to install Diambra CLI directly..."
    
    # Try to install directly into venv bin directory
    pip install --upgrade diambra
    
    # Add the venv bin directory to PATH for this script
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    
    # Check again
    if ! command -v diambra &> /dev/null; then
        echo "${red}${bold}ERROR: Diambra CLI installation failed. Cannot continue.${normal}"
        exit 1
    fi
fi

echo "${green}${bold}Diambra CLI is installed: $(diambra --version 2>&1 || echo 'version info not available')${normal}"
echo "${green}${bold}Dependencies installed successfully!${normal}"

# Step 3: Check for ROM file
echo
echo "${bold}STEP 3: Checking ROM file${normal}"
echo "==================================================="

# Create directory if it doesn't exist
mkdir -p "$PROJECT_DIR/roms"

# Check if ROM file exists
if [ ! -f "$PROJECT_DIR/roms/sfiii3n.zip" ]; then
    echo "${yellow}${bold}ROM file not found: $PROJECT_DIR/roms/sfiii3n.zip${normal}"
    echo "Please copy the Street Fighter III ROM (sfiii3n.zip) to the roms folder."
    echo "You may need to copy it from Windows to WSL."
    echo "In Windows Explorer, you can access the WSL file system by typing: \\\\wsl$"
    echo
    read -p "Press Enter once you've added the ROM file..."
    
    # Check again after user input
    if [ ! -f "$PROJECT_DIR/roms/sfiii3n.zip" ]; then
        echo "${red}${bold}ROM file still not found. Cannot continue.${normal}"
        exit 1
    fi
else
    echo "${green}ROM file found!${normal}"
fi

# Step 4: Create configuration file
echo
echo "${bold}STEP 4: Creating configuration file${normal}"
echo "==================================================="

# Create directory if it doesn't exist
mkdir -p "$PROJECT_DIR/cfg_files/sfiii3n"

# Create configuration file if it doesn't exist
if [ ! -f "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml" ]; then
    echo "Creating configuration file..."
    cat > "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml" << EOF
folders:
  parent_dir: "./results/"
  model_name: "sr6_128x4_das_nc"

settings:
  game_id: "sfiii3n"
  step_ratio: 6
  frame_shape: [128, 128, 0]
  continue_game: 0.0
  action_space: "discrete"
  characters: "Ryu"
  difficulty: 4
  outfits: 1

wrappers_settings:
  normalize_reward: true
  no_attack_buttons_combinations: true
  stack_frames: 4
  dilation: 1
  add_last_action: true
  stack_actions: 12
  scale: true
  exclude_image_scaling: true
  role_relative: true
  flatten: true
  filter_keys: ["action", "own_health", "opp_health", "own_side", "opp_side", "opp_character", "stage", "timer"]

policy_kwargs:
  net_arch: [64, 64]

ppo_settings:
  gamma: 0.94
  model_checkpoint: "0"
  learning_rate: [2.5e-4, 2.5e-6]
  clip_range: [0.15, 0.025]
  batch_size: 256
  n_epochs: 4
  n_steps: 128
  autosave_freq: 512
  time_steps: 1024
EOF
    echo "${green}Configuration file created!${normal}"
else
    echo "Configuration file already exists."
fi

# Step 5: Check if model exists or needs training
echo
echo "${bold}STEP 5: Checking for trained model${normal}"
echo "==================================================="

if [ -f "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip" ]; then
    echo "${green}Found existing trained model!${normal}"
    echo "Location: $PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip"
    
    read -p "Do you want to use this existing model for submission? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "You've chosen to train a new model."
        TRAIN_NEW=true
    else
        TRAIN_NEW=false
    fi
else
    echo "${yellow}No trained model found. Need to train a new model.${normal}"
    TRAIN_NEW=true
fi

# Step 6: Train model if needed
if [ "$TRAIN_NEW" = true ]; then
    echo
    echo "${bold}STEP 6: Training a new model${normal}"
    echo "==================================================="
    
    # Create training script
    echo "Creating training script..."
    cat > "$PROJECT_DIR/training.py" << 'EOF'
import os
import sys
import yaml
import argparse
import numpy as np
from stable_baselines3 import PPO
import gym
import diambra.arena
from diambra.arena.stable_baselines3.wrappers import SB3Wrapper

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

    # Create environment
    env = SB3Wrapper(
        game=cfg["settings"]["game_id"],
        characters=cfg["settings"]["characters"],
        roles=["P1"],
        frame_shape=cfg["settings"]["frame_shape"],
        step_ratio=cfg["settings"]["step_ratio"],
        difficulty=cfg["settings"]["difficulty"]
    )

    # Create model
    if cfg["ppo_settings"].get("model_checkpoint", "0") != "0":
        model = PPO.load(cfg["ppo_settings"]["model_checkpoint"], env=env)
        print(f"Loaded model from {cfg['ppo_settings']['model_checkpoint']}")
    else:
        model = PPO(
            "CnnPolicy",
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
            device="auto"
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
EOF
    
    # Ask for parallel environments
    echo "How many parallel environments do you want to use for training?"
    echo "Higher numbers = faster training but more system resources"
    echo "Recommended: 1-4 for most systems, up to 8 for high-end systems"
    read -p "Enter number (1-8): " parallelEnvs
    
    # Validate input
    if ! [[ "$parallelEnvs" =~ ^[0-9]+$ ]]; then
        parallelEnvs=1
    fi
    
    if [ "$parallelEnvs" -lt 1 ]; then
        parallelEnvs=1
    fi
    
    if [ "$parallelEnvs" -gt 8 ]; then
        parallelEnvs=8
    fi
    
    echo "Starting training with $parallelEnvs parallel environments..."
    echo "This will take some time. Progress will be displayed..."
    
    # Ensure diambra is in the PATH
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    
    if [ "$parallelEnvs" -le 1 ]; then
        diambra run -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
    else
        diambra run -s "$parallelEnvs" -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
    fi
    
    # Check if training was successful
    if [ ! -f "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip" ]; then
        echo "${red}${bold}ERROR: Training failed. Model file not created.${normal}"
        exit 1
    else
        echo "${green}${bold}Training completed successfully!${normal}"
    fi
else
    echo
    echo "${bold}STEP 6: Using existing model (training skipped)${normal}"
    echo "==================================================="
fi

# Step 7: Submit agent
echo
echo "${bold}STEP 7: Submitting agent to Diambra${normal}"
echo "==================================================="

# Ensure diambra is in the PATH
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Check Diambra CLI again
if ! command -v diambra &> /dev/null; then
    echo "${red}${bold}ERROR: Diambra CLI is not available.${normal}"
    echo "Trying to use full path..."
    DIAMBRA_CLI="$VIRTUAL_ENV/bin/diambra"
    if [ ! -f "$DIAMBRA_CLI" ]; then
        echo "${red}${bold}ERROR: Diambra CLI not found at expected location. Cannot continue.${normal}"
        exit 1
    fi
else
    DIAMBRA_CLI="diambra"
fi

# Check Diambra login
echo "Checking Diambra login..."
$DIAMBRA_CLI user info &> /dev/null
if [ $? -ne 0 ]; then
    echo "Logging in to Diambra..."
    $DIAMBRA_CLI user login
fi

# Create submission directory
echo "Preparing submission..."
mkdir -p "$PROJECT_DIR/output/models"

# Copy model file
echo "Copying model file to submission directory..."
cp "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/1022.zip" "$PROJECT_DIR/output/models/1022.zip"

# Change to submission directory
cd "$PROJECT_DIR/output/models"

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << EOF
stable-baselines3
torch
numpy==1.23
EOF

# Initialize agent
echo "Initializing agent..."
$DIAMBRA_CLI agent init .

# Create unique version
VERSION="v$(date +%Y%m%d%H%M)"
echo "Using version tag: $VERSION"

# Submit agent
echo "Submitting agent to Diambra..."
$DIAMBRA_CLI agent submit --submission.difficulty hard --version $VERSION .

# Return to project directory
cd "$PROJECT_DIR"

echo
echo "${green}${bold}Diambra agent setup and submission complete!${normal}"
echo "You can view your submission status on the Diambra website."