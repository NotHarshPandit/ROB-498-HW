import matplotlib.pyplot as plt
import numpy as np
from ddp import DDPController
from tqdm import tqdm
import os
from numpngw import write_apng

def ddp_cartpole(env):
    print("Starting DDP control")
    start_state = np.array([0., np.pi, np.pi, 0., 0., 0.]) # fully down
    goal_state = np.array([0., 0., 0., 0., 0., 0.])
    env.reset(start_state)
    state = start_state.copy()
    goal_state = np.zeros(6)
    time_horizon = 10
    ddp_hyperparams = {'epsilon': 1e-3,
                       'max_iters': 100,
                       'horizon': 20,
                       'backtrack_max_iters': 10,
                       'decay': 0.5,
                       'error_Q': np.diag([0.05, 1, 1, 0.1, 0.1, 0.1])
                       }
    controller = DDPController(start_state, goal_state, ddp_hyperparams)
    states_env = []
    states_controller = []
    error = []
    frames = []
    num_steps = 150
    error_threshold = 0.25
    lowest_error = None
    pbar = tqdm(range(num_steps))
    states_env.append(start_state)
    for _ in pbar:
        state_controller,action = controller.control()
        states_controller.append(state_controller)
        state_env = env.step(action)
        state_env[0] = np.clip(state_env[0], -5, 5) # x position
        states_env.append(state_env)
        img = env.render()
        frames.append(img)
        error_i = controller.calculate_error()
        error.append(error_i)
        if lowest_error is None or error_i < lowest_error:
            lowest_error = error_i
        pbar.set_description(f'Goal Error: {error_i:.4f}, Lowest Error: {lowest_error:.4f}')
        if error_i < error_threshold:
            print("Break")
            break
    
    write_apng(os.path.join(os.getcwd(), "fig", "ddp_cartpole.gif"), frames, delay=100)
    fig, axes = plt.subplots(3, 2, figsize=(8, 8))
    states = np.array(states_controller)
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
    output_file = os.path.join(os.getcwd(), "fig","DDP_Double_Inverted_Pendulum.pdf")
    plt.savefig(output_file)
    plt.close()

    error = np.array(error)
    plt.plot(error)
    plt.title('Goal Error')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    output_file = os.path.join(os.getcwd(), "fig","Cart_Error_ddp.pdf")
    plt.savefig(output_file)
    return 0