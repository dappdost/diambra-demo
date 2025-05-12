#!/bin/bash

# Text formatting
bold=$(tput bold)
normal=$(tput sgr0)
green=$(tput setaf 2)
red=$(tput setaf 1)
yellow=$(tput setaf 3)

echo "${bold}==================================================="
echo "Diambra Street Fighter III AI Agent - All-in-One Setup"
echo "===================================================${normal}"

# Get current directory
PROJECT_DIR=$(pwd)
echo "Using project directory: $PROJECT_DIR"

# Function to check command success
check_command() {
    if [ $? -ne 0 ]; then
        echo "${red}${bold}Error: $1${normal}"
        exit 1
    fi
}

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if grep -q Microsoft /proc/version 2>/dev/null; then
        OS_TYPE="WSL"
    else
        OS_TYPE="Linux"
    fi
else
    echo "${red}${bold}Unsupported OS detected. This script supports macOS, Linux, and WSL.${normal}"
    exit 1
fi

echo "${yellow}Detected OS: $OS_TYPE${normal}"

# Step 1: Install Python 3.9 and dependencies
echo
echo "${bold}STEP 1: Installing Python 3.9 and dependencies${normal}"
echo "==================================================="

if [[ "$OS_TYPE" == "macOS" ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "${yellow}Homebrew is not installed. Installing Homebrew...${normal}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for this session if it's not already
        if [[ "$(uname -m)" == "arm64" ]]; then
            # Apple Silicon
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            # Intel
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    else
        echo "${green}Homebrew is already installed.${normal}"
    fi

    # Update Homebrew
    echo "Updating Homebrew..."
    brew update

    # Install Python 3.9
    echo "Installing Python 3.9..."
    brew install python@3.9

    # Check if Python 3.9 was installed
    if ! command -v python3.9 &> /dev/null; then
        echo "${red}${bold}ERROR: Python 3.9 installation failed.${normal}"
        exit 1
    else
        echo "${green}Python 3.9 installed successfully: $(python3.9 --version)${normal}"
    fi
    
    # Check if Docker is installed and running
    echo "Checking if Docker is installed and running..."
    if ! command -v docker &> /dev/null; then
        echo "${yellow}Docker is not installed. Installing Docker...${normal}"
        brew install --cask docker
        
        echo "${yellow}Please open Docker Desktop and complete the setup process.${normal}"
        echo "Once Docker is running, press Enter to continue..."
        open -a Docker
        read -p ""
    else
        # Check if Docker is running
        if ! docker info &> /dev/null; then
            echo "${yellow}Docker is installed but not running.${normal}"
            echo "Starting Docker..."
            open -a Docker
            
            # Wait for Docker to start
            echo "Waiting for Docker to start..."
            while ! docker info &> /dev/null; do
                echo -n "."
                sleep 2
            done
            echo
            echo "${green}Docker is now running.${normal}"
        else
            echo "${green}Docker is installed and running.${normal}"
        fi
    fi
    
    # Define Python command
    PYTHON_CMD="python3.9"
    
elif [[ "$OS_TYPE" == "Linux" || "$OS_TYPE" == "WSL" ]]; then
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

    # Check if Docker is available
    echo "Checking Docker availability..."
    if ! command -v docker &> /dev/null; then
        echo "${yellow}Docker is not installed.${normal}"
        
        if [[ "$OS_TYPE" == "WSL" ]]; then
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
            echo "Installing Docker..."
            sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
            sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
            sudo apt update
            sudo apt install -y docker-ce
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            
            # Check if docker was installed successfully
            if ! command -v docker &> /dev/null; then
                echo "${red}${bold}ERROR: Docker installation failed.${normal}"
                exit 1
            else
                echo "${green}Docker installed successfully: $(docker --version)${normal}"
                echo "${yellow}NOTE: You may need to log out and log back in for Docker permissions to take effect.${normal}"
            fi
        fi
    else
        echo "${green}Docker is installed: $(docker --version)${normal}"
    fi
    
    # Define Python command
    PYTHON_CMD="python3.9"
fi

# Step 2: Create and activate virtual environment
echo
echo "${bold}STEP 2: Creating Python 3.9 virtual environment${normal}"
echo "==================================================="

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment with Python 3.9..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "${red}${bold}ERROR: Failed to create virtual environment.${normal}"
        echo "Trying to create with system packages..."
        $PYTHON_CMD -m venv venv --system-site-packages
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
            $PYTHON_CMD -m venv venv
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
python -m pip install wheel setuptools
python -m pip install numpy==1.23
python -m pip install "gym<0.27.0,>=0.21.0"
python -m pip install torch pyyaml
python -m pip install diambra
python -m pip install diambra-arena
python -m pip install "diambra-arena[stable-baselines3]"

# Verify critical packages are installed
echo "Verifying key packages..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || { echo "${red}${bold}ERROR: NumPy installation failed.${normal}"; exit 1; }
python -c "import gym; print(f'Gym version: {gym.__version__}')" || { echo "${red}${bold}ERROR: Gym installation failed.${normal}"; exit 1; }
python -c "import diambra.arena; print('Diambra Arena installed')" || { echo "${red}${bold}ERROR: Diambra Arena installation failed.${normal}"; exit 1; }
python -c "import stable_baselines3; print(f'Stable Baselines 3 version: {stable_baselines3.__version__}')" || { echo "${red}${bold}ERROR: Stable Baselines 3 installation failed.${normal}"; exit 1; }
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
        echo "${red}${bold}ERROR: Diambra CLI installation failed.${normal}"
        
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

# Determine how to call diambra
if command -v diambra &> /dev/null; then
    DIAMBRA_CLI="diambra"
else
    DIAMBRA_CLI="$VIRTUAL_ENV/bin/diambra"
fi

echo "${green}${bold}Diambra CLI is installed: $($DIAMBRA_CLI --version 2>&1 || echo 'version info not available')${normal}"
echo "${green}${bold}Dependencies installed successfully!${normal}"

# Step 3: Prepare project files
echo
echo "${bold}STEP 3: Preparing project files${normal}"
echo "==================================================="

# Create necessary directories
mkdir -p "$PROJECT_DIR/roms"
mkdir -p "$PROJECT_DIR/cfg_files/sfiii3n"
mkdir -p "$PROJECT_DIR/output/models"

# Check if ROM file exists
if [ ! -f "$PROJECT_DIR/roms/sfiii3n.zip" ]; then
    echo "${yellow}${bold}ROM file not found: $PROJECT_DIR/roms/sfiii3n.zip${normal}"
    echo "Please copy the Street Fighter III ROM (sfiii3n.zip) to the roms folder."
    
    if [[ "$OS_TYPE" == "WSL" ]]; then
        echo "In Windows Explorer, you can access the WSL file system by typing: \\\\wsl$"
    fi
    
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

# Create configuration file
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

# Create gist.py
echo "Creating gist.py..."
cat > "$PROJECT_DIR/gist.py" << 'EOF'
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
EOF

# Create training.py
echo "Creating training.py..."
cat > "$PROJECT_DIR/training.py" << 'EOF'
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
EOF

# Create evaluate.py
echo "Creating evaluate.py..."
cat > "$PROJECT_DIR/evaluate.py" << 'EOF'
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
EOF

echo "${green}${bold}All necessary files created!${normal}"

# Step 4: Check Diambra login status
echo
echo "${bold}STEP 4: Checking Diambra login status${normal}"
echo "==================================================="

# Check if user is logged in
echo "Checking Diambra login status..."
$DIAMBRA_CLI user info &> /dev/null
if [ $? -ne 0 ]; then
    echo "Please log in to your Diambra account:"
    $DIAMBRA_CLI user login
else
    echo "${green}Already logged in to Diambra account.${normal}"
fi

# Step 5: Menu for actions
echo
echo "${bold}STEP 5: Choose an action${normal}"
echo "==================================================="
echo "1. Test setup with random agent (quick)"
echo "2. Train agent (takes time)"
echo "3. Evaluate trained agent"
echo "4. Submit agent"
echo "5. Do everything in sequence"
echo "6. Exit"
echo

read -p "Enter your choice (1-6): " ACTION_CHOICE

case $ACTION_CHOICE in
    1)
        echo
        echo "${bold}Testing setup with random agent...${normal}"
        $DIAMBRA_CLI run -r "$PROJECT_DIR/roms" python gist.py
        ;;
    2)
        echo
        echo "${bold}Starting agent training...${normal}"
        echo "How many parallel environments do you want to use for training?"
        echo "Higher numbers = faster training but more system resources"
        echo "Recommended: 1-4 for most systems, up to 8 for high-end systems"
        read -p "Enter number of parallel environments (1-8): " parallelEnvs
        
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
        echo "This may take several hours. Progress will be displayed..."
        
        if [ "$parallelEnvs" -le 1 ]; then
            $DIAMBRA_CLI run -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
        else
            $DIAMBRA_CLI run -s "$parallelEnvs" -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
        fi
        ;;
    3)
        echo
        echo "${bold}Evaluating trained agent...${normal}"
        
        if [ ! -f "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip" ]; then
            echo "${red}ERROR: Model file not found!${normal}"
            echo "Please make sure you have trained the agent first."
            echo "Expected model path: $PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip"
        else
            $DIAMBRA_CLI run -r "$PROJECT_DIR/roms" python evaluate.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml" --modelFile "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip"
        fi
        ;;
    4)
        echo
        echo "${bold}Submitting agent to Diambra...${normal}"
        
        if [ ! -f "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip" ]; then
            echo "${red}ERROR: Model file not found!${normal}"
            echo "Please make sure you have trained the agent first."
            echo "Expected model path: $PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip"
        else
            # Copy model file to submission directory
            echo "Copying model file to submission directory..."
            cp "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip" "$PROJECT_DIR/output/models/model.zip"
            
            # Change to models directory
            cd "$PROJECT_DIR/output/models"
            
            # Create requirements.txt
            echo "Creating requirements.txt..."
            echo "stable-baselines3" > requirements.txt
            echo "torch" >> requirements.txt
            echo "numpy==1.23" >> requirements.txt
            
            # Initialize agent
            echo "Initializing agent..."
            $DIAMBRA_CLI agent init .
            
            # Create unique version
            VERSION="v$(date +%Y%m%d%H%M)"
            echo "Using version tag: $VERSION"
            
            # Submit agent
            echo "Submitting agent to Diambra..."
            $DIAMBRA_CLI agent submit --submission.difficulty hard --version "$VERSION" .
            
            # Return to project directory
            cd "$PROJECT_DIR"
        fi
        ;;
    5)
        echo
        echo "${bold}Starting complete process (test, train, evaluate, submit)...${normal}"
        
        # Test with random agent
        echo
        echo "${bold}Step 1: Testing setup with random agent...${normal}"
        $DIAMBRA_CLI run -r "$PROJECT_DIR/roms" python gist.py
        
        # Train agent
        echo
        echo "${bold}Step 2: Training agent...${normal}"
        echo "How many parallel environments do you want to use for training?"
        echo "Higher numbers = faster training but more system resources"
        echo "Recommended: 1-4 for most systems, up to 8 for high-end systems"
        read -p "Enter number of parallel environments (1-8): " parallelEnvs
        
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
        echo "This may take several hours. Progress will be displayed..."
        
        if [ "$parallelEnvs" -le 1 ]; then
            $DIAMBRA_CLI run -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
        else
            $DIAMBRA_CLI run -s "$parallelEnvs" -r "$PROJECT_DIR/roms" python training.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
        fi
        
        # Check if training was successful
        if [ ! -f "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip" ]; then
            echo "${red}ERROR: Training failed. Model file not created.${normal}"
            exit 1
        fi
        
        # Evaluate agent
        echo
        echo "${bold}Step 3: Evaluating trained agent...${normal}"
        $DIAMBRA_CLI run -r "$PROJECT_DIR/roms" python evaluate.py --cfgFile "$PROJECT_DIR/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml" --modelFile "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip"
        
        # Submit agent
        echo
        echo "${bold}Step 4: Submitting agent to Diambra...${normal}"
        
        # Copy model file to submission directory
        echo "Copying model file to submission directory..."
        cp "$PROJECT_DIR/results/sfiii3n/sr6_128x4_das_nc/model/model.zip" "$PROJECT_DIR/output/models/model.zip"
        
        # Change to models directory
        cd "$PROJECT_DIR/output/models"
        
        # Create requirements.txt
        echo "Creating requirements.txt..."
        echo "stable-baselines3"
        # Create requirements.txt
        echo "Creating requirements.txt..."
        echo "stable-baselines3" > requirements.txt
        echo "torch" >> requirements.txt
        echo "numpy==1.23" >> requirements.txt
        
        # Initialize agent
        echo "Initializing agent..."
        $DIAMBRA_CLI agent init .
        
        # Create unique version
        VERSION="v$(date +%Y%m%d%H%M)"
        echo "Using version tag: $VERSION"
        
        # Submit agent
        echo "Submitting agent to Diambra..."
        $DIAMBRA_CLI agent submit --submission.difficulty hard --version "$VERSION" .
        
        # Return to project directory
        cd "$PROJECT_DIR"
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "${red}Invalid choice. Please run the script again and select a valid option.${normal}"
        exit 1
        ;;
esac

echo
echo "${green}${bold}Process completed!${normal}"
echo "You can view your submission status on the Diambra website."