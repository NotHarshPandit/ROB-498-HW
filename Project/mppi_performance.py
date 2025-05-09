import matplotlib.pyplot as plt
import numpy as np
from mppi_control import MPPIController, get_cartpole_mppi_hyperparams
import torch
from tqdm import tqdm
import os
from numpngw import write_apng

def mppi_cartpole(env):
    print("Starting MPPI control")
    start_state = np.array([0, np.pi,np.pi,0, 0, 0], dtype=np.float32) + np.random.rand(6,)
    env.reset(start_state)  
    state = start_state.copy()
    goal_state = np.zeros(6)
    controller = MPPIController(env, num_samples=500, horizon=30, hyperparams=get_cartpole_mppi_hyperparams())
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)
    num_steps = 500
    error = []
    states = []
    frames = []
    pbar = tqdm(range(num_steps))
    lowest_error = None
    for _ in pbar:
        state = torch.tensor(state, dtype=torch.float32)
        control = controller.command(state)
        state = env.step(control)
        states.append(state)
        error_i = np.linalg.norm(state-goal_state[:7])
        error.append(error_i)
        img = env.render()
        frames.append(img)
        pbar.set_description(f'Goal Error: {error_i:.4f}')
        if lowest_error is None or error_i < lowest_error:
            lowest_error = error_i
        pbar.set_description(f'Goal Error: {error_i:.4f}, Lowest Error: {lowest_error:.4f}')
        if error_i < .5:
            break
    write_apng(os.path.join(os.getcwd(), "fig", "mppi_cartpole.gif"), frames, delay=100)
    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
    states = np.array(states)
    goal_state = np.array(goal_state).reshape(1, 6)
    goal_state = np.repeat(goal_state, states.shape[0], axis=0)
    axes[0][0].plot(states[:, 0])
    axes[0][0].plot(goal_state[:,0], '--')
    axes[0][0].title.set_text('x')

    axes[1][0].plot(states[:, 1])
    axes[1][0].plot(goal_state[:,1], '--')
    axes[1][0].title.set_text('theta1')

    axes[2][0].plot(states[:, 2])
    axes[2][0].plot(goal_state[:,2], '--')
    axes[2][0].title.set_text('theta2')

    axes[0][1].plot(states[:, 3])
    axes[0][1].plot(goal_state[:,3], '--')
    axes[0][1].title.set_text('x_dot')


    axes[1][1].plot(states[:, 4])
    axes[1][1].plot(goal_state[:,4], '--')
    axes[1][1].title.set_text('theta1_dot')

    axes[2][1].plot(states[:, 5])
    axes[2][1].plot(goal_state[:,5], '--')
    axes[2][1].title.set_text('theta2_dot')
    axes[2][1].legend(['actual', 'goal'])
    output_file = os.path.join(os.getcwd(), "fig","Cart_States_MPPI.pdf")
    plt.savefig(output_file)
    plt.close()


    error = np.array(error)
    plt.plot(error)
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    output_file = os.path.join(os.getcwd(), "fig","Cart_Error_MPPI.pdf")
    plt.savefig(output_file)
    return 0
    