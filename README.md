# MyFirstAgent: Diambra AI Agent for Street Fighter III

This repository contains code for training and deploying an AI agent for Street Fighter III using Diambra Arena and Reinforcement Learning.

## Project Overview

This AI agent uses Reinforcement Learning (specifically PPO from Stable Baselines 3) to learn how to play Street Fighter III. The training process involves the agent playing multiple matches against the computer opponent to develop combat strategies and skills.

## Setup Instructions

### Prerequisites

- Python 3.9.7
- NumPy 1.23
- Docker Desktop
- Street Fighter III 3rd Strike ROM (sfiii3n.zip)

### Installation Steps

1. **Install Docker Desktop**
   - Download and install from [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
   - Ensure Docker is running before proceeding

2. **Install Python Dependencies**
   ```bash
   python -m pip install --upgrade pip
   pip install numpy==1.23
   python -m pip install diambra
   python -m pip install diambra-arena
   pip install diambra-arena[stable-baselines3]
   ```

3. **ROM Setup**
   - Create a `roms` folder in your project directory
   - Place the sfiii3n.zip file in this folder

4. **Create Diambra Account**
   - Sign up at [https://diambra.ai/](https://diambra.ai/)

## Project Structure

```
MyFirstAgent/
├── roms/
│   └── sfiii3n.zip
├── cfg_files/
│   └── sfiii3n/
│       └── sr6_128x4_das_nc.yaml
├── gist.py
├── training.py
├── evaluate.py
├── submissionagent.py
├── submission-manifest.yaml
├── output/
│-- requirements.txt
└── README.md
```

## Usage Instructions

### 1. Running a Random Agent

To test your setup with a basic random agent:

```bash
diambra run -r /path/to/roms python gist.py
```

### 2. Training Your Agent

To start training your AI agent:

```bash
diambra run -r /path/to/roms python training.py --cfgFile /path/to/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml
```

For parallel training environments (faster training):

```bash
diambra run -s 4 -r /path/to/roms python training.py --cfgFile /path/to/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml
```

### 3. Evaluating Your Agent

After training, evaluate your agent's performance:

```bash
diambra run -r /path/to/roms python evaluate.py --cfgFile /path/to/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml --modelFile /path/to/results/sfiii3n/sr6_128x4_das_nc/model/model.zip
```

### 4. Submitting Your Agent

To submit your agent to Diambra:



```bash
diambra agent init ./output/models/
```

```bash
cd ./output/models/
```
```bash
pip install -r requirements.txt
```
```bash
diambra agent submit --submission.difficulty hard .

```
## Configuration Options

The agent's behavior and training parameters can be customized in the `cfg_files/sfiii3n/sr6_128x4_das_nc.yaml` file:

- **Character Selection**: Change the `characters` field (e.g., from "Ryu" to "Ken")
- **Training Duration**: Adjust `time_steps` for longer/shorter training
- **Checkpoint Frequency**: Modify `autosave_freq` to control how often models are saved
- **Learning Parameters**: Tune `batch_size`, `n_epochs`, and `n_steps` for different learning behaviors

## Training Tips

1. For best results, train for at least 48 hours (significantly more time_steps than the default)
2. Keep backups of significant milestone checkpoints
3. To continue training from a previous checkpoint, update the `model_checkpoint` value in the config file

## Troubleshooting

- Ensure Docker Desktop is running before executing any commands
- Verify the ROM file is correctly placed in the roms directory
- Check paths in your commands match your actual directory structure

## Resources

- [Diambra Documentation](https://docs.diambra.ai/)
- [Diambra Discord](https://discord.com/invite/diambra)
- [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/)

## License

[Your chosen license]