import numpy as np
import torch 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from cartpole_env import *
from numpngw import write_apng
from IPython.display import Image



def normalize_angle_batch(angles):
    normalized = np.mod(angles, 2 * np.pi)  # Wrap to [0, 2π)
    normalized = np.where(normalized > np.pi, normalized - 2 * np.pi, normalized)  # Map to [-π, π)
    return normalized


def check_anayltics(env):
    np.random.seed(2)
    print("Plotting analytics vs pybullet")
    # first let's generate a random control sequence
    T = 100
    control_sequence = np.random.randn(T, 1).astype(np.float32)
    # control_sequence = np.ones((T, 1)).astype(np.float32)

    #actual angle from vertical line
    start_state = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    env.reset(start_state)
    frames=[] #frames to create animated png
    frames.append(env.render()) 
    for i in tqdm(range(100)):
        action = env.action_space.sample()
        s = env.step(action)
        img = env.render()
        frames.append(img)
    output_file = os.path.join(os.getcwd(), "fig","cartpole_example.gif")
    write_apng(output_file, frames, delay=10)

    # We use the simulator to simulate a trajectory
    states_pybullet = np.zeros((T+1, 6))
    states_pybullet[0,:] = start_state
    frames = []
    for t in range(T):
        states_pybullet[t+1,:] = env.step(control_sequence[t])
        img = env.render()
        frames.append(img)

    # print('states_pybullet: ', states_pybullet.shape)

    current_state = torch.from_numpy(start_state).reshape(1, 6)
    current_control = torch.from_numpy(control_sequence[0]).reshape(1, 1) # add batch dimension to control

    # Now we will use your analytic dynamics to simulate a trajectory
    states_analytic = torch.zeros(T+1,6) 
    states_analytic[0,:] = torch.from_numpy(start_state).reshape(1,6)
    for t in range(T):
        current_state = states_analytic[t]
        
        current_control = torch.from_numpy(control_sequence[t]).reshape(1, 1) # add batch dimension to control   
        # print("current_state: ", current_state.shape) 
        states_analytic[t+1,:] = env.dynamics_analytic(state = current_state, action = current_control)
        
    # convert back to numpy for plotting
    states_analytic = states_analytic.reshape(T+1, 6).numpy()

    # states_analytic = -states_analytic
    # states_analytic = np.zeros_like(states_analytic)

    output_file = os.path.join(os.getcwd(), "fig", "cartpole_example.gif")
    write_apng(output_file, frames, delay=10)
    Image(filename=output_file)

    # Plot and compare - They should be indistinguishable 
    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
    plt.tight_layout()
    axes[0][0].plot(states_analytic[:, 0], label='analytic')
    axes[0][0].plot(states_pybullet[:, 0], '--', label='pybullet')
    axes[0][0].title.set_text('x')

    axes[1][0].plot(states_analytic[:, 1])
    axes[1][0].plot(states_pybullet[:, 1], '--')
    axes[1][0].title.set_text('theta_1')

    axes[2][0].plot(states_analytic[:, 2])
    axes[2][0].plot(states_pybullet[:, 2], '--')
    axes[2][0].title.set_text('theta_2')

    axes[0][1].plot(states_analytic[:, 3])
    axes[0][1].plot(states_pybullet[:, 3], '--')
    axes[0][1].title.set_text('x_dot')

    axes[1][1].plot(states_analytic[:, 4])
    axes[1][1].plot(states_pybullet[:, 4], '--')
    axes[1][1].title.set_text('theta_1_dot')

    axes[2][1].plot(states_analytic[:, 5])
    axes[2][1].plot(states_pybullet[:, 5], '--')
    axes[2][1].title.set_text('theta_2_dot')

    axes[0][0].legend()
    output_file = os.path.join(os.getcwd(), "fig", "cartpole_analytic_vs_pybullet.png")
    plt.savefig(output_file)
    return None

