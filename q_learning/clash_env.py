import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
from stable_baselines3 import DQN

# Import your vision script!
import screen_detection as sd 

class ClashRoyaleEnv(gym.Env):
    def __init__(self):
        super(ClashRoyaleEnv, self).__init__()

        # OBSERVATION SPACE: We must downscale the massive phone screen to 84x84 for the CNN.
        # We keep 3 channels (RGB)
        # Now we also add elixir information into the observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "elixir": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)
        })
        
        # Track previous health to calculate damage deltas for rewards
        self.prev_blue_health = 1.0
        self.prev_red_health = 1.0

        # Card/arena coordinates borrowed from random_agent_adb.py
        self.card_slots_x = [470, 720, 980, 1250]
        self.card_slot_y = 2300
        self.arena_x_min = 150
        self.arena_x_max = 1300
        self.arena_y_min = 150 # allow placement near the top of the arena, this gets unlocked after tower destruction for troops, but spells can always go anywhere
        self.arena_y_max = 1880
        self.grid_rows = 16 # 32 tiles long by 18 tiles wide is the full size of the arena
        self.grid_cols = 9
        self.tap_delay = 0.2
        # Discretize the gamespace grid
        self.zones = self.grid_rows * self.grid_cols # 16*9 = 144 zones, coarse grid approach to improve training speed
        # 4 cards * 144 zones = 576 discrete actions + 1 "Do nothing" action = 577 total actions
        self.actions = self.zones * len(self.card_slots_x) + 1 # +1 for "Do nothing" action
        self.action_space = spaces.Discrete(self.actions)

        # Store the state of the princess towers in class variables
        self.first_red_dest = False
        self.second_red_dest = False # enemy towers
        # our towers
        self.first_blue_dest = False
        self.second_blue_dest = False

    def step(self, action):
            # 1. Initialize required variables to prevent crash
            elix_zero = False # only do an action with elixir available, helps with game state as well
            reward = 0.0
            done = False
            info = {}

            # GET THE CURRENT STATE
            raw_rgb = sd.get_screen_rgb() # check the screen twice is time expensive, but we need this to create a good gameplay loop
            # Add a separate check for the elixir level at 0, potential end game signal
            if sd.read_elixir_bar(raw_rgb) == 0:
                print("No elixir detected, possible end game state. Checking win/loss conditions again to confirm.")
                time.sleep(0.1) # Wait a moment for the end game screen to update
                elix_zero = True # update our status
                action = 0

            # EXECUTE ACTION  (The agent chose this based on the PREVIOUS step's observation)
            if action != 0: 
                self._execute_adb_tap(action)
                
            # Wait for the game to register the tap and animations to play
            time.sleep(0.1)

            # Get new state after action is executed and the game has updated
            raw_rgb = sd.get_screen_rgb()
            obs_image = cv2.resize(raw_rgb, (84, 84))

            # 4. CHECK IF GAME IS OVER
            win_status = sd.check_win_condition(raw_rgb)
            info["elixir"] = sd.read_elixir_bar(raw_rgb)
            obs_elixir = np.array([info["elixir"]], dtype=np.float32)

            obs = {
                "image": obs_image,
                "elixir": obs_elixir
            }
            
            if win_status == "Win":
                reward = 3000.0  # Massive bonus for winning, bumped up big time (x3) for better reward shaping
                done = True
                return obs, reward, done, False, info
            elif win_status == "Loss":
                reward = -3000.0 # Massive penalty for losing
                done = True
                return obs, reward, done, False, info

            # only calculate ongoing reward if the game is still in progress and we have elixir to work with, otherwise we might be calculating deltas during the end game screen which could be noisy and not meaningful    
            if not done and not elix_zero: 
                # 5. CALCULATE ONGOING REWARD (Using Deltas)
                blue_health_list, red_health_list = sd.check_tower_health(raw_rgb, "both")
                # extract the red and blue princess towers for reward shaping
                first_red_p_tower  = red_health_list[0]
                second_red_p_tower = red_health_list[1] # dont need to get king tower cuz thats the win condition
                # blue towers
                first_blue_p_tower = blue_health_list[0]
                second_blue_p_tower = blue_health_list[1]

                # Now check to see if any are destroyed, and implement reward shaping
                if first_red_p_tower <= 0.01 and not self.first_red_dest:
                    reward += 750.0 # big reward for destroying a tower
                    self.first_red_dest = True # now set to True so we don't repeatedly reward for a destroyed tower
                # Second red tower
                if second_red_p_tower <= 0.01 and not self.second_red_dest:
                    reward += 750.0 # big reward for destroying a tower
                    self.second_red_dest = True # now set to True so we don't repeatedly reward for a destroyed tower

                # Now check if any of our towers are destroyed
                if first_blue_p_tower <= 0.01 and not self.first_blue_dest:
                    reward -= 750.0 # big reward loss for losing a tower
                    self.first_blue_dest = True # now set to True so we don't repeatedly reward for a destroyed tower
                if second_blue_p_tower <= 0.01 and not self.second_blue_dest:
                    reward -= 750.0
                    self.second_blue_dest = True
                
                # now comibine all the tower healths into one for the rest of our reward shaping (health difference)
                blue_health = ((first_blue_p_tower+second_blue_p_tower)/3.6) + blue_health_list[2]*1.6/3.6 # use our old estimation to recall the total health
                red_health = ((first_red_p_tower+second_red_p_tower)/3.6) + red_health_list[2]*1.6/3.6
                # ensure no accidental tower healing via troop walking in place of destroyed tower
                blue_health = min(blue_health, self.prev_blue_health)
                red_health = min(red_health, self.prev_red_health)
                
                # Delta is (Current Health - Previous Health)
                # If we took damage, blue_delta is negative.
                blue_delta = blue_health - self.prev_blue_health 
                
                # If enemy took damage, red_delta is negative.
                red_delta = red_health - self.prev_red_health 
                
                # Formula: (Our Health Kept) - (Their Health Kept)
                # Example: We take 0.1 damage (blue_delta = -0.1). They take 0.2 damage (red_delta = -0.2).
                # (-0.1) - (-0.2) = +0.1 net positive reward!
                reward += (blue_delta - red_delta) * 100
                if blue_delta - red_delta == 0:
                    reward += 50.0 # small reward for not losing damage, or defending properly
                
                # 6. Update trackers for the NEXT step
                self.prev_blue_health = blue_health
                self.prev_red_health = red_health
            return obs, reward, done, False, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        print("Match ended. Resetting environment...")
        
        # 1. Tap the "OK" button on the win/loss banner
        # (You will need to find the exact X Y coordinates using your pixel_inspector)
        sd.device.shell("input tap 800 2250") # Replace with real OK button coords
        sd.device.shell("input tap 725 2250") # double tap to secure right ok button
        print("waiting for menu to load...")
        time.sleep(6) # Wait for menu to load
        
        # 2. Tap the "Battle" button on the main menu
        sd.device.shell("input tap 750 2000") # Replace with real Battle button coords
        
        # 3. Wait for matchmaking and the loading screen to finish
        print("Waiting for new match to load...")
        time.sleep(6) 
        
        # Reset health trackers for the new game
        self.prev_blue_health = 1.0
        self.prev_red_health = 1.0 # intialize at full health

        # Reset tower statuses
        self.first_blue_dest = False
        self.second_blue_dest = False
        # Red
        self.first_red_dest = False
        self.second_red_dest = False
        
        # Grab the first frame of the new game
        raw_rgb = sd.get_screen_rgb()
        obs_image = cv2.resize(raw_rgb, (84, 84)) # redefined for our observation dictionary

        # matches start with 5 elixir so reset our state to that. if it's wrong, it will get corrected fairly soon
        obs_elixir = np.array([5.0], dtype=np.float32)

        # now put our observation spaces into one obs dictionary
        obs = {
            "image": obs_image,
            "elixir": obs_elixir
        }
        return obs, {}

    def _execute_adb_tap(self, action):
        """Translates a non-zero action into card selection + arena placement taps."""
        if action <= 0 or action >= self.action_space.n:
            return

        # action 1..2304 -> 4 cards x (32*18) placement cells
        action_idx = action - 1 # Shift down by 1 to make it zero-indexed
        cells_per_card = self.zones # 32*18 = 576
        card_idx = action_idx // cells_per_card # Determine which card to play (0-3)
        cell_idx = action_idx % cells_per_card # Determine which cell in the grid to place the card (0-575)

        card_idx = int(np.clip(card_idx, 0, len(self.card_slots_x) - 1)) # Ensure card index is valid
        row = cell_idx // self.grid_cols # Determine row in the grid
        col = cell_idx % self.grid_cols # Determine column in the grid

        x_step = (self.arena_x_max - self.arena_x_min) / (self.grid_cols - 1) # Calculate step size for columns
        y_step = (self.arena_y_max - self.arena_y_min) / (self.grid_rows - 1) # Calculate step size for rows
        arena_x = int(self.arena_x_min + col * x_step) # Calculate X coordinate for the tap based on the column
        arena_y = int(self.arena_y_min + row * y_step) # Calculate Y coordinate for the tap based on the row

        sd.device.shell(f"input tap {self.card_slots_x[card_idx]} {self.card_slot_y}") # Tap the card in hand
        time.sleep(self.tap_delay) # Wait for game UI to register
        sd.device.shell(f"input tap {arena_x} {arena_y}") # Tap the arena to deploy the card