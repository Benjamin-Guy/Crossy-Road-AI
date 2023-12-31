import pyautogui
import pygetwindow as gw
import time
import random
from PIL import Image
import pytesseract
import re
import numpy as np
import cv2

def get_score(image):
    # Preprocess the image for better OCR accuracy
    # Convert to grayscale
    gray_image = image.convert('L')
    # Apply thresholding
    _, thresh_image = cv2.threshold(np.array(gray_image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Resize the image
    resized_image = Image.fromarray(thresh_image).resize((2 * gray_image.width, 2 * gray_image.height), Image.Resampling.BILINEAR)
    
    # Use pytesseract to extract the score
    text = pytesseract.image_to_string(resized_image, config='--psm 6')
    
    # Extract the score from the text using regex
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    else:
        return None

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

             # Halve the resolution of the image
            new_size = (screenshot_width // 4, screenshot_height // 4)
            resized_image = grayscale_image.resize(new_size)

            # Save or process the resized screenshot
            resized_image.save(f"screenshot_{int(time.time())}.png")

            # Wait for the next frame
            time.sleep(interval)
            break # This is for testing purposes.

# Capture the screen of the BlueStacks window every 0.1 seconds for 0.5 seconds
capture_bluestacks_screen(0.1, 0.5)