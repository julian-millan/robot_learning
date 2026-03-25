# PPO Policy Gradient Training

Trains a Proximal Policy Optimization (PPO) agent to play Clash Royale on a Bluestacks Android emulator via ADB.

## How It Works

### Overview

The script uses [Stable Baselines3](https://stable-baselines3.readthedocs.io/)'s PPO implementation with a `CnnPolicy` — a convolutional neural network that takes raw game screenshots as input and outputs card placement actions.

### Environment

The agent interacts with `ClashRoyaleEnv` (defined in `q_learning/clash_env.py`), a custom Gymnasium environment that:

- **Observation space**: 84x84 RGB screenshots of the game, captured via ADB screencap
- **Action space**: 2305 discrete actions — 4 cards × 576 arena grid cells (32 rows × 18 cols) + 1 "do nothing" action
- **Reward**: shaped by health deltas each step; +1000 for a win, -1000 for a loss

### Startup Sequence

Before training begins, `ppo_train.py` automatically:

1. Starts the ADB server (`adb start-server`)
2. Opens Bluestacks (`open -a Bluestacks`) and waits 10 seconds for it to load
3. Launches Clash Royale via ADB monkey input
4. `screen_detection.py` connects to Bluestacks at `127.0.0.1:5555` and holds a reference to the ADB device for all subsequent screen captures and taps

### Training Loop

Each PPO step:
1. Captures the current screen via ADB screencap and resizes it to 84x84
2. Checks elixir level and win/loss condition
3. Translates the chosen action index into a card tap + arena placement tap via ADB
4. Captures the next screen and computes the reward from tower health deltas

The agent checkpoints every 250 steps to `./models/` and logs to TensorBoard under `./clash_tensorboard/`.

### Model Persistence

On startup, the script attempts to load `ppo_clash_royale_agent_final.zip` to resume a previous run. If not found, it initializes a fresh PPO agent. After training, the model is saved back to `ppo_clash_royale_agent_final.zip`.

## Running

From the **project root**:

```bash
python -m policy_gradient.ppo_train
```

Make sure Bluestacks is open and ADB is enabled in Bluestacks settings (Settings → Advanced → Enable ADB) before running.

## Monitoring Training

```bash
tensorboard --logdir ./clash_tensorboard/
```
