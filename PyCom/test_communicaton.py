import subprocess
import time
import logging
from mlagents_envs.environment import UnityEnvironment

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Path to your Unity executable (update with your actual path)
unity_executable_path = "C:/Users/prebe/Unity/AI/MLAgents/Builds/UnityEnvironment.exe"

# Run Unity executable as a subprocess
try:
    logging.info("Starting Unity Environment...")
    #unity_process = subprocess.Popen(unity_executable_path)
    
    # Wait for a few seconds to ensure Unity starts properly
    time.sleep(5)
    # Try to connect to Unity Environment using ML-Agents
    env = UnityEnvironment(file_name=None)
    
    logging.info("Unity Environment connected successfully!")
    # Environment must be reset to ensure proper communication between python and Unity
    env_info = env.reset()
    print(env_info)
    # Check available behavior names
    behavior_names = list(env.behavior_specs.keys())
    logging.info(f"Available Behavior Names: {behavior_names}")
    
    # If behaviors are available, pick the first one
    if behavior_names:
        behavior_name = behavior_names[0]
        logging.info(f"Using Behavior: {behavior_name}")
        behavior_spec = env.behavior_specs[behavior_name]
        logging.info(f"Behavior Spec: {behavior_spec}")
    else:
        logging.warning("No behaviors found!")

    brain_names = env.brain_names
    logging.info("Available brains: {brain_names}")
    
except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    # Make sure to close the Unity environment
    if 'env' in locals():
        env.close()
    if 'unity_process' in locals():
        unity_process.terminate()
        logging.info("Unity process terminated.")
