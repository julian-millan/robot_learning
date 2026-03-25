
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
sys.path.append('..')
import subprocess
import time

# Start ADB server if not running
try:
    result = subprocess.run(['adb', 'start-server'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("ADB server started successfully.")
    else:
        print(f"Failed to start ADB: {result.stderr}")
except FileNotFoundError:
    print("ADB not found. Ensure Android SDK platform-tools are installed and in PATH.")
except subprocess.TimeoutExpired:
    print("ADB start timed out.")

# Wait a moment for ADB to initialize
time.sleep(2)

# Always open Bluestacks (brings to front if already running)
try:
    print("Opening Bluestacks...")
    subprocess.run(['open', '-a', 'Bluestacks'])
    # Wait for Bluestacks to start or come to front
    time.sleep(10)
    print("Bluestacks is ready.")
except Exception as e:
    print(f"Error opening Bluestacks: {e}")


# Launch Clash Royale app via ADB (always attempt after opening Bluestacks)
try:
    # Get list of devices
    devices_result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=5)
    if 'device' in devices_result.stdout and 'emulator' in devices_result.stdout:
        print("Device connected. Launching Clash Royale...")
        launch_result = subprocess.run(['adb', 'shell', 'monkey', '-p', 'com.supercell.clashroyale', '-c', 'android.intent.category.LAUNCHER', '1'], capture_output=True, text=True, timeout=10)
        if launch_result.returncode == 0:
            print("Clash Royale launched successfully.")
            time.sleep(5)  # Wait for app to load
        else:
            print(f"Failed to launch Clash Royale: {launch_result.stderr}")
    else:
        print("No ADB device found. Ensure Bluestacks is connected.")
except subprocess.TimeoutExpired:
    print("ADB command timed out.")
except Exception as e:
    print(f"Error launching Clash Royale: {e}")


from q_learning.clash_env import ClashRoyaleEnv

import time

'''
run `python -m policy_gradient.ppo_train`

'''

env = ClashRoyaleEnv()

print("Loading saved agent...")
try:
    model = PPO.load(
        "./policy_gradient/ppo_clash_royale_agent_final",
        env=env,
        tensorboard_log="./policy_gradient/clash_tensorboard/"
    )
    print("Agent loaded successfully!")
except Exception as e:
    print(f"Failed to load saved agent. Reason: {e}")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./policy_gradient/clash_tensorboard/"
    )

checkpoint_callback = CheckpointCallback(
    save_freq=250,
    save_path='./policy_gradient/models/',
    name_prefix='clash_ppo'
)

print("Resuming training. Ensure Bluestacks is open and in a match.")

current_time = time.strftime("%Y%m%d-%H%M")
custom_name = f"PPO_Run_{current_time}"

model.learn(
    total_timesteps=10000,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
    log_interval=1,
    tb_log_name=custom_name,
    progress_bar=True
)

model.save("./policy_gradient/ppo_clash_royale_agent_final")
print("Training saved successfully!")