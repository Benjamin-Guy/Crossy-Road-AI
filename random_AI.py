import pyautogui
import time
import random
import pygetwindow as gw
import cv2
import pytesseract
from PIL import Image
import re
import numpy as np
import os

# List of possible actions (WASD keys and spacebar)
actions = ['w', 'a', 's', 'd']

# Configure the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_random_action():
    # Choose a random action from the list
    action = random.choice(actions)
    
    # Perform the action
    pyautogui.keyDown(action)
    time.sleep(0.1)  # Hold the key for a short duration
    pyautogui.keyUp(action)

def start_game():
    bluestacks_window = gw.getWindowsWithTitle('BlueStacks App Player')[0]
    window_center = (bluestacks_window.left + bluestacks_window.width / 2,
                        bluestacks_window.top + bluestacks_window.height / 2)
    pyautogui.click(window_center)

def game_over():
    # Find the game over icon on the screen
    game_over_icon_location = pyautogui.locateCenterOnScreen('game_over.png', confidence=0.9)
    if game_over_icon_location is not None:
        # Press space to restart the game
        time.sleep(2)
        pyautogui.click(game_over_icon_location)
        return True
    return False

def get_score(image):
    # Convert the PIL Image to a NumPy array and then to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dictionary to hold matches for each number
    matches = {}
    
    # Iterate over each number
    for number in range(10):
        # Load all templates for the current number
        templates = [cv2.imread(f'numbers/{number}{("_" + str(i) if i != 0 else "")}.png', 0) for i in range(len([name for name in os.listdir('numbers') if name.startswith(str(number))]))]

        for template in templates:
            w, h = template.shape[::-1]

            # Perform template matching
            res = cv2.matchTemplate(thresh_image, template, cv2.TM_CCOEFF_NORMED)
            
            # Set a threshold for matching
            threshold = 0.98
            loc = np.where(res >= threshold)
            
            # Store the matches
            for pt in zip(*loc[::-1]):
                # Check if this match is close to an existing match for the same number
                close_to_existing = any(abs(pt[0] - existing_x) < w for existing_x in matches.keys() if matches[existing_x] == str(number))
                if not close_to_existing:
                    matches[pt[0]] = str(number)  # Store the number as a string, keyed by its x-coordinate

    # Combine the numbers based on their x-coordinate
    score = ''.join(matches[x] for x in sorted(matches))

    return int(score) if score else None
    
def capture_score_region(bluestacks_window):
    # Define the region where the score is displayed (top-left corner)
    # This example assumes the score region is one eighth the width and height of the total window size
    
    score_region_width = bluestacks_window.width // 9
    score_region_height = bluestacks_window.height // 5
    score_region = (bluestacks_window.left, bluestacks_window.top + 33, score_region_width, score_region_height - 33)

    # Capture the screen region of the BlueStacks window where the score is displayed
    score_screenshot = pyautogui.screenshot(region=score_region)

    # Convert the screenshot to an OpenCV image
    score_screenshot_cv = cv2.cvtColor(np.array(score_screenshot), cv2.COLOR_RGB2BGR)

    # Define the range for white color
    lower_white = np.array([250, 250, 250], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Threshold the image to get only white colors
    mask = cv2.inRange(score_screenshot_cv, lower_white, upper_white)
    result = cv2.bitwise_and(score_screenshot_cv, score_screenshot_cv, mask=mask)

    # Convert back to PIL Image
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    # Save the processed score image
    #result_image.save(f"score_screenshot_{int(time.time())}.png")

    return result_image

def main():
    print("Starting the game initially.")
    bluestacks_window = gw.getWindowsWithTitle('BlueStacks App Player')[0]
    bluestacks_window.activate()
    start_game()

    print("Started playing.")
    while True:
        if game_over():
            print("-   I have died. Restarting.")
            # Wait a bit for the game to go back to the start screen
            time.sleep(2)
            start_game()
        else:
            print("I am still alive. Continuing.")
            # Perform a random action
            #perform_random_action()
            score_image = capture_score_region(bluestacks_window)
            score = get_score(score_image)
            print("Current score:", score)
        
        # Wait a short period before the next action
        time.sleep(0.2)

if __name__ == "__main__":
    main()