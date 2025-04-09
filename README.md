# Reinforcement Learning in Unity

This project explores **reinforcement learning** in a Unity environment, where a car learns to navigate the Spielberg racing track with reinforcement learning. Using Unity's **ML-Agents Release 22**, different algorithms compete to get the best lap time.  

## Overview  

The goal is to train an agent to drive efficiently around the track using deep reinforcement learning. The project evaluates four different approaches:  

- **PPO (Proximal Policy Optimization)** – ML-Agents built-in  
- **SAC (Soft Actor-Critic)** – ML-Agents built-in  
- **DDPG (Deep Deterministic Policy Gradient)** – Custom implementation  
- **DQN (Deep Q-Network)** – Custom implementation  

🔹 Custom implementations are located in the `/PyCom` folder.  

## Demo  

https://github.com/user-attachments/assets/e36eb867-eaee-4a62-9e5b-f46832c522dc


## Installation and setup

Follow the installation guide from [ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation.md).

Navigate to the PyCom folder and run the following command to train with custom algorithm

```bash
python perform_actions_ddpg.py
```
To use a ML-Agents algorithm run

```bash
mlagents-learn config/sac/car.yaml --run-id=first_sac_run
```

After running the commands the Unity environment must start. Click the play button in the Unity editor to start the environment.

After training the weights can be tested with 

```bash
python test_ddpg.py
```

or by assigning the onnx-file to the agent component in the Unity Inspector.

## Project Structure  

```bash
/UnityProject          # Unity environment with ML-Agents
/PyCom                 # Custom implementations and weights (DQN, DDPG)
/results               # Tensorboard graphs
README.md              # Project documentation
requirements.txt       # Python dependencies
```

## Analysis

![sac_ppo](https://github.com/user-attachments/assets/f73bc912-ce91-4a70-8eb8-4223ecc26e7f)

SAC can learn quickly, but PPO still gets a better lap time.
