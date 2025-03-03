import logging
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from learning import *

logging.basicConfig(level=logging.DEBUG)

# Unity build path
unity_executable_path = "C:/Users/prebe/Unity/AI/MLAgents/Builds/UnityEnvironment.exe"
# Configuration
channel = EngineConfigurationChannel()

try:
    logging.info("Starting Unity Environment...")

    ## Opening a Unity build instead or an editor
    #unity_process = subprocess.Popen(unity_executable_path)

    ## Wait for a few seconds to ensure Unity starts properly
    # time.sleep(5)


    # Connect to ML-Agents Unity Environment in th Unity editor
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    
    logging.info("Unity Environment connected successfully!")
    
    # Reset environment
    env.reset()
    
    # Check available behaviors
    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        logging.warning("No behaviors found!")
        exit()

    behavior_name = behavior_names[0]
    logging.info(f"Using Behavior: {behavior_name}")
    behavior_spec = env.behavior_specs[behavior_name]
    logging.info(f"Behavior Spec: {behavior_spec}")

    # Get action space dimensions
    continuous_action_size = behavior_spec.action_spec.continuous_size
    discrete_action_size = behavior_spec.action_spec.discrete_size

    logging.info(f"Continuous Action Size: {continuous_action_size}")
    logging.info(f"Discrete Action Size: {discrete_action_size}")

    # Main interaction loop
    run_episodes(env, behavior_name, continuous_action_size, discrete_action_size)

except Exception as e:
    logging.error(f"Connection error: {e}")

finally:
    # Closing environment
    if 'env' in locals():
        env.close()
    #if 'unity_process' in locals():
        #unity_process.terminate()
        #logging.info("Unity process terminated.")



