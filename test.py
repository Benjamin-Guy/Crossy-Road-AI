import pyautogui
import pygetwindow as gw
import time
import random
from PIL import Image

def capture_bluestacks_screen(interval, duration):
    # Find the BlueStacks window
    bluestacks_window = gw.getWindowsWithTitle('BlueStacks App Player')[0]
    bluestacks_window.activate()
    
    if bluestacks_window is not None:
        start_time = time.time()

        while (time.time() - start_time) < duration:
            # Bring the BlueStacks window to the front
            bluestacks_window.activate()

            # Capture the screen region of the BlueStacks window
            screenshot = pyautogui.screenshot(region=bluestacks_window.box)

            # Determine the width and height of the screenshot
            screenshot_width, screenshot_height = screenshot.size
            crop_area = (0, 33, screenshot_width - 33, screenshot_height)

            # Crop the image to the specified area
            cropped_image = screenshot.crop(crop_area)

            # Convert the image to grayscale
            grayscale_image = cropped_image.convert('L')

            # Save or process the grayscale screenshot
            grayscale_image.save(f"screenshot_{int(time.time())}.png")

            # Wait for the next frame
            time.sleep(interval)
            break # This is for testing purposes.

# Capture the screen of the BlueStacks window every 0.1 seconds for 0.5 seconds
capture_bluestacks_screen(0.1, 0.5)