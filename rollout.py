from IPython.display import clear_output
import matplotlib.pyplot as plt
from gym.wrappers import Monitor
from IPython.display import HTML
import base64
import shutil


def rollout(agent, env, render=True):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.predict(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if render:
            screen = env.render(mode='rgb_array')
            clear_output(wait=True)
            img = plt.imshow(screen)
            plt.axis('off')
            plt.show()

        state = next_state

    return total_reward

def save_rollout_video(agent, env_name, video_filename, render=True):
    wrapped_env = Monitor(gym.make(env_name), video_filename, force=True)
    
    state = wrapped_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.predict(state)
        state, reward, done, _ = wrapped_env.step(action)
        total_reward += reward
        if render:
            wrapped_env.render()
    
    wrapped_env.close()
    return total_reward


def display_video(video_filename):
    video = open(video_filename, "rb").read()
    video_encoded = base64.b64encode(video).decode("ascii")
    video_tag = f'<video width="640" height="480" controls alt="rollout video"><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4" /></video>'
    return HTML(video_tag)
