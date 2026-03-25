from ppadb.client import Client as AdbClient
from PIL import Image
import numpy as np
import io
import time
import cv2


print("Starting Screen Detection for Bluestacks Emulator")
import subprocess
subprocess.run(['adb', 'connect', '127.0.0.1:5555'], capture_output=True, text=True)
client = AdbClient(host="127.0.0.1", port=5037)
print("Connected to ADB server at 127.0.0.1:5037")
devices = client.devices()
print('Connected devices:', devices)

if len(devices) == 0:
    print("No Bluestacks instance found. Make sure it's open and ADB is enabled.")
    exit()
device = devices[0]

red_health = np.array([226, 38, 94])  # Example RGB for red health bar
blue_health = np.array([114, 210, 255])  # Example RGB for blue health bar
elixir_color = np.array([240, 137, 244])  # Example RGB for elixir bar
banner_color = np.array([38, 99, 176])  # Example RGB for win/lose banner
win_color = np.array([102, 255, 255])  # Example RGB for win banner
lose_color = np.array([255, 204, 255])  # Example RGB for lose banner
crown_color = np.array([231, 193, 58])  # Example RGB for crown icon on win banner
hit_color = np.array([178, 178, 179]) # Example RGB for hit marker on tower health bars

banner_pixel = (726, 1442)  # Example pixel location for win/lose banner
win_pixel = (721, 1036)  # Example pixel location for win banner
lose_pixel = (685, 305)  # Example pixel location for lose banner
my_crown_pixel = (722, 1917)  # Example pixel location for crown icon on win banner
enemy_crown_pixel = (722, 63)  # Example pixel location for crown icon on lose banner

def get_screen_rgb():
    """Capture the screen RGB values from the device.

    Returns:
        np.ndarray: The RGB values of the screen as a 3D numpy array (height x width x 3).
    """
    # Capture screenshot from device
    raw = device.screencap()

    # Convert to image
    image = Image.open(io.BytesIO(raw))

    # Convert to numpy array (H x W x 4)
    rgb = np.array(image)
    rgb = rgb[:, :, :3]  # Keep only the first 3 channels (RGB, ignore alpha)

    return rgb

def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb = param
        print(x,y,rgb[y,x])
        
import cv2

def pixel_inspector(rgb):
    """Inspect the pixel values of the screen.

    Args:
        rgb (np.ndarray): The RGB values of the screen as a 3D numpy array (height x width x 3).
    """

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    display = img_bgr.copy()

    def mouse_move(event, x, y, flags, param):
        nonlocal display

        display = img_bgr.copy()

        if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
            r, g, b = rgb[y, x]

            text = f"x:{x} y:{y} RGB:{r},{g},{b}"

            cv2.putText(
                display,
                text,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1,
                cv2.LINE_AA
            )

            cv2.circle(display, (x,y), 3, (0,255,0), -1)

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"CLICK -> ({x},{y}) RGB:{rgb[y,x]}")

    cv2.namedWindow("Pixel Inspector")
    cv2.setMouseCallback("Pixel Inspector", mouse_move)

    # while True:
    #     cv2.imshow("Pixel Inspector", display)

    #     key = cv2.waitKey(20) & 0xFF

    #     if key == ord('q') or key == 27:
    #         break

    #     if cv2.getWindowProperty("Pixel Inspector", cv2.WND_PROP_VISIBLE) < 1:
    #         break

    cv2.destroyAllWindows()

def read_elixir_bar(rgb):
    """Read the elixir bar level from the screen.

    Args:
        rgb (np.ndarray): The RGB values of the screen as a 3D numpy array (height x width x 3).

    Returns:
        int: The elixir level (0-10).
    """
    # Example: Check a specific region for elixir bar color
    # This is just a placeholder and may need to be adjusted based on the actual game UI
    elixir_region = rgb[2480, 540:1400:100, :]  # Adjust these coordinates as needed
    elixir_region = np.vstack((rgb[2480,480], elixir_region)) # have to add the first elixir region separately
    count = 0
    for pixel in elixir_region:
        if check_pixel_color(pixel, elixir_color, 75):  count += 1
    
    return count

def check_win_condition(rgb):
    """Check the win condition based on the screen RGB values.

    Args:
        rgb (np.ndarray): The RGB values of the screen as a 3D numpy array (height x width x 3).

    Returns:
        str: "Win", "Loss", or "In Progress" based on the game state.
    """
    if check_pixel_color(rgb[banner_pixel[1], banner_pixel[0]], banner_color):
        if check_pixel_color(rgb[win_pixel[1], win_pixel[0]], win_color):
            return "Win"
        elif check_pixel_color(rgb[lose_pixel[1], lose_pixel[0]], lose_color):
            return "Loss"
        else:
            return "Unknown Result"
    else:
        return "In Progress"
    
def check_pixel_color(pixel_rgb, target_rgb, tolerance=30):
    """Check if a pixel's color matches a target color within a tolerance.

    Args:
        pixel_rgb (np.ndarray): The RGB values of the pixel to check.
        target_rgb (np.ndarray): The RGB values of the target color.
        tolerance (int, optional): The color matching tolerance. Defaults to 20.

    Returns:
        bool: True if the pixel color matches the target color within the tolerance, False otherwise.
    """
    return all(abs(int(pixel_rgb[i]) - target_rgb[i]) <= tolerance for i in range(3))

def check_tower_health(rgb, tower_color):
    """Check the health of a tower based on its color.

    Args:
        rgb (np.ndarray): The RGB values of the screen as a 3D numpy array (height x width x 3).
        tower_color (string): Which tower to check ("red", "blue" or "both").

    Returns:
        tuple: The health percentage heuristic of the tower (0.0 to 1.0).
        If "both" is selected, returns a tuple with both blue and red tower health percentages.
    """
    if tower_color == "red":
        bar_color = red_health
        princess_region = rgb[390, 288:440, :]  # Adjust these coordinates as needed
        princess_region = np.vstack((princess_region, rgb[390, 1043:1197, :])) # combine both princess towers
        crown_pixel = rgb[enemy_crown_pixel[1], enemy_crown_pixel[0]]
        crown_region = rgb[100,645:854,:]
    elif tower_color == "blue":
        bar_color = blue_health
        princess_region = rgb[1590, 288:440, :]  # Adjust these coordinates as needed
        princess_region = np.vstack((princess_region, rgb[1590, 1043:1197, :])) # combine both princess towers
        crown_pixel = rgb[my_crown_pixel[1], my_crown_pixel[0]]
        crown_region = rgb[1934,645:854,:]
    elif tower_color == "both":
        red_tower_health = check_tower_health(rgb, "red")
        blue_tower_health = check_tower_health(rgb, "blue")
        return blue_tower_health,red_tower_health
    else:
        raise ValueError("Invalid tower color. Choose 'red', 'blue', or 'both'.")

    total_pixels =  princess_region.shape[0] # number of pixels in the health region (both princess towers combined)
    princess_percentage = sum(check_pixel_color(princess_region[i], bar_color) or check_pixel_color(princess_region[i], hit_color)
                              for i in range(total_pixels))/ total_pixels

    if check_pixel_color(crown_pixel, crown_color):
        #print("Crown detected, assuming full health for king tower")    
        crown_percentage = 1.0
    else:
        crown_percentage = sum(check_pixel_color(crown_region[i], bar_color) or check_pixel_color(crown_region[i], hit_color)
                               for i in range(crown_region.shape[0]))/ crown_region.shape[0]
    #print(f"Princess tower health percentage: {princess_percentage:.2f}, Crown region health percentage: {crown_percentage:.2f}")
    return (princess_percentage*2/3.6 + crown_percentage*1.6/3.6) # scaled assuming princess towers are roughly 60% of king tower


def is_in_match(rgb):
    """Returns True if we're currently in a live battle (elixir bar is visible)."""
    return read_elixir_bar(rgb) > 0


def main():
    """Main entry point of the program."""
    print("Starting screen detection. Press Ctrl+C to stop.")
    # Call other functions from here
    # result = some_other_function()
    # print(result)
    # Run the agent in a loop
    try:
        rgb = get_screen_rgb()
        elixir_level = read_elixir_bar(rgb)
        win_status = check_win_condition(rgb)
        print(f"Elixir level: {elixir_level}")
        print(f"Win status: {win_status}")
        health = check_tower_health(rgb, "both")
        print(f"Red Tower health: {health[1]:.2f}, Blue Tower health: {health[0]:.2f}")
        # pixel_inspector(rgb)
    except KeyboardInterrupt:
        print("Agent stopped.")


if __name__ == "__main__":
    main()
