import pyautogui
import time
import random
import pygetwindow as gw
import cv2

# List of possible actions (WASD keys and spacebar)
actions = ['w', 'a', 's', 'd']

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
            perform_random_action()
        
        # Wait a short period before the next action
        time.sleep(0.1)

if __name__ == "__main__":
    main()