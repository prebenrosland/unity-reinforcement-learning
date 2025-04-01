# Reinforcement Learning in Unity

This project explores reinforcement learning in a Unity environment, where an AI-controlled car learns to navigate the Spielberg racing track. Using Unity's ML-Agents Release 22, we compare different RL algorithms, including built-in implementations and custom models.

The goal is to train an agent to drive efficiently around the track using reinforcement learning. The project evaluates different approaches:

- **PPO (Proximal Policy Optimization)** – ML-Agents built-in  
- **SAC (Soft Actor-Critic)** – ML-Agents built-in  
- **DDPG (Deep Deterministic Policy Gradient)** – Custom implementation  
- **DQN (Deep Q-Network)** – Custom implementation  

Custom implementations are located in the `/PyCom` folder.  

### DDPG Demo

https://github.com/user-attachments/assets/94f5c1df-e14e-4a00-a616-a8ceef2d8ae8

## Installation and setup

Clone the project

```bash
git clone https://github.com/prebenrosland/unity-reinforcement-learning.git
```
Install mlagents and mlagents-env according to official [documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation.md)

Install package dependencies

```bash
pip install torch
pip install numpy
```

Open the project in a Unity environment and add 3D models to the correct folders. The [track](https://sketchfab.com/3d-models/cartoon-race-track-spielberg-23dbb21af64e407286fd16e29c9aea25) and [car](https://syntystore.com/products/polygon-city-pack) should be in Project/Assets/Car/Models.

To run the code of custom implementation (ddpg example)

```bash
python perform_actinos_ddpg_car.py
```
Then click the play button in the Unity Editor

Using an agent with pretrained weights the following command can be run

```bash
python test_run_agent_car.py
```

To use  built-in implementations of PPO or SAC in ML-Agents, refer to .yaml file to determine architecture, episode length and other parameters (ppo example)

```bash
mlagents-learn config/ppo/car.yaml --run-id=my_first_ppo_run
```

## Analysis
![sac_ppo](https://github.com/user-attachments/assets/318642cb-4cfc-4239-9599-d8c0847e2200)

SAC converges significantly faster, however PPO ultimaltely ends up with a faster lap time.

