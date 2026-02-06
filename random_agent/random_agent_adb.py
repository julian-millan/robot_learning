import random
import time
import ppadb
from ppadb.client import Client as AdbClient

print("Starting Random Card Play Agent for Bluestacks Emulator")
# 1. Connect to Bluestacks ADB server
client = AdbClient(host="127.0.0.1", port=5037)  # Default Bluestacks ADB port
devices = client.devices()
print('Connected devices:', devices)

if len(devices) == 0:
    print("No Bluestacks instance found. Make sure it's open and ADB is enabled.")
    exit()

device = devices[0]

def play_random_card():
    # Randomly pick one of the 4 card slots in your hand
    # Card slots are roughly spread horizontally at the bottom
    card = random.randint(1, 4);
    cards_x = {1: 470, 2: 720, 3: 980, 4: 1250}
    card_y = 2300  # Y coordinate for the card hand area
    
    # Randomly pick a spot in the arena (usually the bottom half for defense)
    arena_x = random.randint(150, 1300)
    arena_y = random.randint(1150, 1880)

    print(f"Playing card {card} to arena {arena_x}, {arena_y}")

    # Action: Tap the card
    device.shell(f"input tap {cards_x[card]} {card_y}")
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

def main():
    """Main entry point of the program."""
    print("Starting random card play agent. Press Ctrl+C to stop.")
    # Call other functions from here
    # result = some_other_function()
    # print(result)
    # Run the agent in a loop
    try:
        while True:
            play_random_card()
            # Wait a few seconds between moves so elixir can recharge
            time.sleep(random.uniform(2.0, 5.0))
    except KeyboardInterrupt:
        print("Agent stopped.")

if __name__ == "__main__":
    main()
