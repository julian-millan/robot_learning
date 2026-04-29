from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from clash_env import ClashRoyaleEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack # for frame stacking
import os

import time

# 1. Initialize your custom environment
env = ClashRoyaleEnv()
env = Monitor(env) # Wrap in Monitor for logging
env = DummyVecEnv([lambda: env]) # Wrap in DummyVecEnv for compatibility
env = VecFrameStack(env, n_stack=4) # Stack 4 frames to give the agent temporal context, adjust n_stack as needed based on performance and memory constraints, from Deepmind's work

# 2. LOAD the saved agent instead of creating a new one
print("Loading saved agent...")
# Make sure the filename matches what you saved it as (without the .zip extension)
try:
    model = DQN.load("dqn_clash_elixir_v1", 
                     env=env, tensorboard_log="./clash_elixir_tensorboard/",
                     exploration_initial_eps=1.0, # Start with full exploration when resuming
                     exploration_final_eps=0.02, # End with a small amount of exploration
                       exploration_fraction=0.99, # Explore for 99% of the training time, locking in at 2% exploration for the last 5% of training
                         learning_rate=2.5e-4,
                           batch_size=128,
                           replay_buffer_kwargs={"handle_timeout_termination": False},
                           buffer_size=10000) # a little more epxloration
    print("Agent loaded successfully!")
except Exception as e:
    print(f"Failed to load saved agent. Reason: {e}")
    model = DQN(
        "MultiInputPolicy",           
        env,
        learning_rate=2.5e-4,
        buffer_size=40000,
        exploration_initial_eps=1.0, # Start with full exploration when starting
        exploration_final_eps=0.02, # End with a small amount of exploration     
        exploration_fraction=0.99, 
        verbose=1,
        tensorboard_log="./clash_elixir_tensorboard/",
        batch_size=128,
        replay_buffer_kwargs={"handle_timeout_termination": False} # error conflict with the above line
    )
# 1. Ask Python to figure out exactly where train.py is located on your hard drive
script_directory = os.path.dirname(os.path.abspath(__file__))

# 2. Tell it to make the save folder exactly next to train.py
absolute_save_path = os.path.join(script_directory, 'stacked_models')

# Create the checkpoint callback again so it keeps saving backups
checkpoint_callback = CheckpointCallback(
    save_freq=100,
    save_path=absolute_save_path,
    name_prefix='clash_dqn_elixir'
)

# 3. Continue training!
print("Resuming training. Ensure Bluestacks is open and in a match.")
# reset_num_timesteps=False ensures your logging doesn't restart at step 0

current_time = time.strftime("%Y%m%d-%H%M")
custom_name = f"DQN_Run_{current_time}"
model.learn(
    total_timesteps=6000, 
    callback=checkpoint_callback, 
    reset_num_timesteps=True, 
    log_interval=1,
    tb_log_name=custom_name
) 

# 4. Save the updated brain
model.save("dqn_clash_elixir_v1") # This will overwrite your previous save, so make sure to backup if you want to keep it
print("Training saved successfully!")