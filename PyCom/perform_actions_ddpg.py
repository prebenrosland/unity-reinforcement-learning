import logging
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from torch.utils.tensorboard import SummaryWriter

from Utils.agents import *
from Utils.models import *
from Utils.tools import *


batch_size = 128
n_episodes = 10000
n_batches_train = 1
exp_replay_buffer_size = int(2e5)
epsilon_decay = 0.999
epsilon = 1
epsilon_limit = 0.1
target_network_update_rate = 0.001
discount_factor = 0.99

tensorboard = SummaryWriter("runs/DDPG_Training")

logging.basicConfig(level=logging.DEBUG)

# Configuration
channel = EngineConfigurationChannel()

try:
    logging.info("Starting Unity Environment...")

    # Connect to ML-Agents Unity Environment
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    
    logging.info("Unity Environment connected successfully!")

    # Reset the environment, no need to store return value
    env.reset()

    # Check behaviors (Agent brains)
    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        logging.warning("No behaviors found!")
        exit()

    behavior_name = behavior_names[0]
    logging.info(f"Using Behavior: {behavior_name}")
    behavior_spec = env.behavior_specs[behavior_name]

    # Get action dimensions from agent
    continuous_action_size = behavior_spec.action_spec.continuous_size
    discrete_action_size = behavior_spec.action_spec.discrete_size
    logging.info(f"Continuous Action Size: {continuous_action_size}")
    logging.info(f"Discrete Action Size: {discrete_action_size}")

    # Defining amount of action and state sizes
    state_size = behavior_spec.observation_specs[0].shape[0]
    action_size = continuous_action_size

    list_rewards = []

    # Initializing the agent network
    agent = DDPGAgent(critic=Critic, actor=Actor, state_size=state_size, action_size=action_size, 
                    target_network_update_rate=target_network_update_rate, discount_factor=discount_factor,
                    batch_size=batch_size, buffer_size=exp_replay_buffer_size,
                    num_batches=n_batches_train)
    
    epsilons = []
    score_list = []
    step = 0

    # Loop over episodes
    for episode in range(n_episodes):
        epsilons.append(epsilon)
        epsilon = epsilon_decay * epsilon + (1 - epsilon_decay) * epsilon_limit

        # Reset environment
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        state = decision_steps.obs[0]
        score = 0
        done = [False]

        while not any(done):
            step += 1
            # Agent takes action
            action = agent.act(state, epsilon=epsilon)

            # Create ActionTuple for Unity agent
            action_tuple = ActionTuple(continuous=action)

            # Set actions and step
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # Get steps
            new_decision_steps, new_terminal_steps = env.get_steps(behavior_name)

            # Check termination
            if len(new_terminal_steps) > 0:
                next_state = new_terminal_steps.obs[0]
                reward = new_terminal_steps.reward
                done = [True]
            else:
                next_state = new_decision_steps.obs[0]
                reward = new_decision_steps.reward
                done = [False]

            # Update agent
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += np.sum(reward)

        score_list.append(score)
        tensorboard.add_scalar('Score per Episode', score, episode)
        tensorboard.add_scalar('Epsilon', epsilon, episode)

        if (episode+1) % 10 == 0:
            logging.info(f'\n----- Episode : {episode + 1} ----- Average score : {np.mean(score_list)}, ----- Step : {step} \n')
            score_list = []

        if score > 5000:
            torch.save(agent.critic.state_dict(), 'Weights/car/checkpoint_critic.pth')
            torch.save(agent.critic_target.state_dict(), 'Weights/car/checkpoint_critic_target.pth')
            torch.save(agent.actor.state_dict(), 'Weights/car/checkpoint_actor.pth')
            torch.save(agent.actor_target.state_dict(), 'Weights/car/checkpoint_actor_target.pth')

        if step >= 3600000:
            break
        

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    env.close()