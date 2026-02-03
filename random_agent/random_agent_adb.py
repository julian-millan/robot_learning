import random
import time
import ppadb
from ppadb.client import Client as AdbClient

# 1. Connect to LDPlayer
client = AdbClient(host="127.0.0.1", port=5037)
devices = client.devices()

if len(devices) == 0:
    print("No LDPlayer found. Make sure it's open and ADB is enabled.")
    exit()

device = devices[0]

def play_random_card():
    # Randomly pick one of the 4 card slots in your hand
    # Card slots are roughly spread horizontally at the bottom
    card_x = random.choice([150, 300, 450, 600])
    card_y = 1500
    
    # Randomly pick a spot in the arena (usually the bottom half for defense)
    arena_x = random.randint(100, 650)
    arena_y = random.randint(450, 750)

    print(f"Playing card at {card_x}, {card_y} to arena {arena_x}, {arena_y}")

    # Action: Tap the card
    device.shell(f"input tap {card_x} {card_y}")
    time.sleep(0.2) # Small delay to select
    
    # Action: Tap the arena to deploy
    device.shell(f"input tap {arena_x} {arena_y}")

def get_screen_size(device):
    # Runs 'adb shell wm size' which returns something like "Physical size: 1080x1920"
    result = device.shell("wm size")
    size_str = result.split(":")[-1].strip() # Get the "1080x1920" part
    width, height = map(int, size_str.split("x"))
    return width, height

width, height = get_screen_size(device)
print(f"Emulator Dimensions: Width={width}, Height={height}")

# Run the agent in a loop
try:
    while True:
        play_random_card()
        # Wait a few seconds between moves so elixir can recharge
        time.sleep(random.uniform(2.0, 5.0))
except KeyboardInterrupt:
    print("Agent stopped.")