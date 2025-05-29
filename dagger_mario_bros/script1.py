import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import cv2

# Setup
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = env.reset()
done = False

# Main loop
try:
    while not done:
        # Look ahead at a ground-level region for obstacles
        obstacle_area = obs[180:210, 140:160, :]  # region ahead of Mario
        darkness = np.mean(obstacle_area)
        print(f"Darkness level: {darkness:.2f}")
        # Choose action based on obstacle presence
        if darkness < 160:
            action = 5  # jump
        else:
            action = 1  # run right

        obs, reward, done, info = env.step(action)

        # Optional: draw the area we're watching
        obs_drawable = obs.copy().astype(np.uint8)
        cv2.rectangle(obs_drawable, (140, 180), (160, 210), (0, 0, 255), 1)
        cv2.imshow("Mario Expert", cv2.cvtColor(obs_drawable, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) == 27:  # ESC to exit
            break
finally:
    env.close()
    cv2.destroyAllWindows()