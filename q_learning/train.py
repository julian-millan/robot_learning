from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from clash_env import ClashRoyaleEnv

import time

# 1. Initialize your custom environment
env = ClashRoyaleEnv()

# 2. LOAD the saved agent instead of creating a new one
print("Loading saved agent...")
# Make sure the filename matches what you saved it as (without the .zip extension)
try:
    model = DQN.load("dqn_clash_royale_agent_final", env=env, tensorboard_log="./clash_tensorboard/", exploration_fraction=0.20, learning_rate=1e-4) # a little more epxloration
    print("Agent loaded successfully!")
except Exception as e:
    print(f"Failed to load saved agent. Reason: {e}")
    model = DQN(
        "CnnPolicy",           
        env,
        learning_rate=1e-3,
        buffer_size=10000,     
        exploration_fraction=0.15, 
        verbose=1,
        tensorboard_log="./clash_tensorboard/"
    )

# Create the checkpoint callback again so it keeps saving backups
checkpoint_callback = CheckpointCallback(
    save_freq=250,
    save_path='./models/',
    name_prefix='clash_dqn'
)

# 3. Continue training!
print("Resuming training. Ensure Bluestacks is open and in a match.")
# reset_num_timesteps=False ensures your logging doesn't restart at step 0

current_time = time.strftime("%Y%m%d-%H%M")
custom_name = f"DQN_Run_{current_time}"
model.learn(
    total_timesteps=5000, 
    callback=checkpoint_callback, 
    reset_num_timesteps=False, 
    log_interval=1,
    tb_log_name=custom_name
) 

# 4. Save the updated brain
model.save("dqn_clash_royale_agent_final")
print("Training saved successfully!")