from ppadb.client import Client as AdbClient
from PIL import Image
import numpy as np
import io
import time
import cv2


print("Starting Screen Detection for Bluestacks Emulator")
client = AdbClient(host="127.0.0.1", port=5037)
print("Connected to ADB server at 127.0.0.1:5037")
devices = client.devices()
print('Connected devices:', devices)

if len(devices) == 0:
    print("No Bluestacks instance found. Make sure it's open and ADB is enabled.")
    exit()

device = devices[0]

def get_screen_rgb():
    # Capture screenshot from device
    raw = device.screencap()

    # Convert to image
    image = Image.open(io.BytesIO(raw))

    # Convert to numpy array (H x W x 3)
    rgb = np.array(image)

    return rgb

def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb = param
        print(x,y,rgb[y,x])
        
import cv2

def pixel_inspector(rgb):

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

    while True:
        cv2.imshow("Pixel Inspector", display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            break

        if cv2.getWindowProperty("Pixel Inspector", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def main():
    """Main entry point of the program."""
    print("Starting screen detection. Press Ctrl+C to stop.")
    # Call other functions from here
    # result = some_other_function()
    # print(result)
    # Run the agent in a loop
    try:
        rgb = get_screen_rgb()
        print("Captured screen RGB data. Starting pixel inspector...")
        pixel_inspector(rgb[:, :,:3])  # Pass only RGB channels to the inspector

    except KeyboardInterrupt:
        print("Agent stopped.")


if __name__ == "__main__":
    main()
