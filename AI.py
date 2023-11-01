import pyautogui
import time
import random
import pygetwindow as gw
import cv2
from PIL import Image
import re
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# List of possible actions (WASD keys and spacebar)
actions = ['w', 'a', 's', 'd']

# Hyperparameters
STATE_SHAPE = (4, 208, 363)
ACTION_SIZE = len(actions)
LEARNING_RATE = 0.001
MAX_MEMORY = 100000  # Maximum number of experiences stored in the memory buffer
BATCH_SIZE = 20  # Batch size for training
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TAU = 0.001  # for soft update of target parameters

# Initialize the frame stack with a maximum length of 4
frame_stack = deque(maxlen=4)

class DQNNetwork(nn.Module):
    def __init__(self, state_shape, action_size):
        super(DQNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Use a dummy input to calculate the size of the flattened layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape)
            dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))
            self.flattened_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.flattened_size, 512)  # Adjusted the size to match the flattened layer
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = EPSILON

        self.model = DQNNetwork(state_shape, action_size).cuda()
        self.target_model = DQNNetwork(state_shape, action_size).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32).cuda()
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).cuda()
        actions = torch.tensor(actions).cuda()
        rewards = torch.tensor(rewards).cuda()
        dones = torch.tensor(dones, dtype=torch.float32).cuda()

        target = rewards + (1 - dones) * GAMMA * torch.max(self.target_model(next_states).detach(), dim=1)[0]
        target_f = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(target_f, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

        # Soft update of the target network
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

    def save(self, filename="dqn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="dqn_model.pth"):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set the model to evaluation mode

def preprocess_image(image):
    # Convert the image to grayscale (if not already)
    image = image.convert('L')
    # Convert the image to a numpy array and normalize it
    image = np.array(image) / 255.0
    return image

def update_frame_stack(frame_stack, new_frame):
    # Preprocess the new frame
    new_frame = preprocess_image(new_frame)
    # Append the new frame to the deque
    frame_stack.append(new_frame)
    # Stack the frames along a new dimension to create a single array
    if len(frame_stack) < 4:
        # If we have fewer than 4 frames, repeat the first frame to fill the stack
        return np.stack([frame_stack[0]] * (4 - len(frame_stack)) + list(frame_stack), axis=0)
    else:
        return np.stack(frame_stack, axis=0)

def perform_action(action):
    # Perform the specified action
    print(f"-   I pressed: {action}")
    pyautogui.keyDown(action)
    pyautogui.keyUp(action)

def capture_screenshot():
    bluestacks_window = gw.getWindowsWithTitle('BlueStacks App Player')[0]
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
    #resized_image.save(f"screenshot_.png")

    return resized_image

def calculate_reward(old_score, new_score, game_over_flag):
    if game_over_flag:
        return -100  # Example negative reward for dying
    else:
        return new_score - old_score  # Reward based on score increase

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

# Main Loop for DQN
def main():
    bluestacks_window = gw.getWindowsWithTitle('BlueStacks App Player')[0]
    bluestacks_window.activate()
    time.sleep(2)
    start_game()
    initial_frame = capture_screenshot()
    state = update_frame_stack(frame_stack, initial_frame)
    score = 0
    replay_counter = 0

    while True:
        print(f"This is my {replay_counter} attempt at this game!")
        action = agent.act(state)
        perform_action(actions[action])
        
        next_frame = capture_screenshot()
        next_state = update_frame_stack(frame_stack, next_frame)
        
        score_image = capture_score_region(bluestacks_window)
        old_score = score
        score = get_score(score_image)
        if score is None or score < old_score:
            score = old_score
        print(f"-       I have a score of {score}.")
        
        game_over_flag = game_over()
        
        reward = calculate_reward(old_score, score, game_over_flag)

        agent.remember(state, action, reward, next_state, game_over_flag)
        state = next_state
        
        agent.replay()
        replay_counter += 1

         # Save the model every 10 replays
        if replay_counter % 10 == 0:
            agent.save('dqn_model_latest.pth')
            print("Model saved.")

        if game_over_flag:
            print("-   I have died. Restarting.")
            time.sleep(2)
            start_game()
            initial_frame = capture_screenshot()
            state = update_frame_stack(frame_stack, initial_frame)
            score = 0
            agent.update_target_model()

if __name__ == "__main__":
    agent = DQNAgent(STATE_SHAPE, ACTION_SIZE)

    # Load an existing model if needed
    model_path = "dqn_model_latest.pth"
    if os.path.isfile(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")

    main()