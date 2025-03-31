import logging
import numpy as np
from mlagents_envs.base_env import ActionTuple

steps = 100
episodes = 500


def run_episodes_randomly(env, behavior_name, continuous_action_size, discrete_action_size):
    # Run for x episodes
    for episode in range(episodes):

        # Resetting the environment for each episode
        env.reset()
        logging.info(f"Starting Episode {episode + 1}")

        episode_reward = episode_step_randomly(100, env, behavior_name, continuous_action_size, discrete_action_size)

        logging.info(f"Episode {episode + 1} complete, Reward : {episode_reward}")

def episode_step_randomly(steps, env, behavior_name, continuous_action_size, discrete_action_size):

    episode_reward = 0

    # Run for x steps each episode
    for _ in range(steps):
            # Get observations from envorinment
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Calculating rewards
            step_reward = np.sum(decision_steps.reward) + np.sum(terminal_steps.reward)
            episode_reward += step_reward

            # Select random actions
            num_agents = len(decision_steps)
            continuous_actions = np.random.uniform(-1, 1, (num_agents, continuous_action_size))
            discrete_actions = np.random.randint(0, 2, (num_agents, discrete_action_size))

            action_tuple = ActionTuple()
            action_tuple.add_continuous(continuous_actions)
            if discrete_action_size > 0:
                action_tuple.add_discrete(discrete_actions)

            # Send actions to agent in Unity environment
            env.set_actions(behavior_name, action_tuple)

            env.step()

    return episode_reward / steps