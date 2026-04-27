from torch.utils.tensorboard import SummaryWriter

# 1. Point this to your exact TensorBoard log folder
# (Using the path from your previous screenshots)
log_path = "./clash_stacked_tensorboard/Random_Baseline"
writer = SummaryWriter(log_dir=log_path)

# 2. Set your random agent's average score here!
# (Example: Let's pretend a random agent averages -150 per game)
random_average_score = 0.0

# 3. Draw the flat line across the X-axis (from step 0 to 10,000)
# We log a point every 1,000 steps so TensorBoard draws a continuous line
print(f"Drawing baseline at {random_average_score}...")
for step in range(0, 9000, 1000):
    writer.add_scalar("rollout/ep_rew_mean", random_average_score, step)

writer.close()
print("Success! Refresh your TensorBoard browser tab.")