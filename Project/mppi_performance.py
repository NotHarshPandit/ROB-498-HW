import matplotlib.pyplot as plt
import numpy as np
from mppi_control import MPPIController, get_cartpole_mppi_hyperparams
import torch
from numpngw import write_apng
from tqdm import tqdm

def mppi_cartpole(env):
    start_state = np.array([0, np.pi,np.pi,0, 0, 0], dtype=np.float32) + np.random.rand(6,)
     # 6D state vector (x, theta1, theta2, x_dot, theta1_dot, theta2_dot)
    env.reset(start_state)  
    state = start_state.copy()
    goal_state = np.zeros(6)
    controller = MPPIController(env, num_samples=500, horizon=30, hyperparams=get_cartpole_mppi_hyperparams())
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)
    frames = []
    num_steps = 500
    error = []
    states = []
    pbar = tqdm(range(num_steps))
    for i in pbar:
        state = torch.tensor(state, dtype=torch.float32)
        control = controller.command(state)
        state = env.step(control)
        states.append(state)
        error_i = np.linalg.norm(state-goal_state[:7])
        error.append(error_i)
        # print("i, error_i", i, error_i)
        pbar.set_description(f'Goal Error: {error_i:.4f}')
        img = env.render()
        frames.append(img)
        if error_i < .8:
            break
    write_apng("./fig/MPPI Double Inverted Pendulum.gif", frames, delay=10)
    print("creating animated gif, please wait")
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
    plt.savefig('./fig/Cart_States_MPPI.png')
    plt.close()


    error = np.array(error)
    plt.plot(error)
    plt.title('Goal Error')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.savefig('./fig/Cart_Error_MPPI.png')
    return 0
    