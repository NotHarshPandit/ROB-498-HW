import matplotlib.pyplot as plt
import numpy as np
from iLQG import ILQGController
from numpngw import write_apng
from tqdm import tqdm
import torch

def iLQG_cartpole(env):
    start_state = np.array([0, np.pi,np.pi,0, 0, 0], dtype=np.float32) + np.random.rand(6,)
    state = start_state.copy()
    goal_state = np.zeros(6)
    time_horizon = 100
    controller = ILQGController(
          umax = 30.0,
          state_dim = 6,
          pred_time=time_horizon)
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)
    u_seq = np.random.randn(time_horizon, 1).astype(np.float32)
    x_seq = [state.copy()]
    for t in range(time_horizon):
        x_seq.append(env.dynamics_analytic_np(x_seq[-1], u_seq[t]))
    num_steps = 150
    env.reset(start_state)
    error = []
    states = []
    frames = []
    pbar = tqdm(range(num_steps))
    for i in pbar:
        k_seq, kk_seq = controller.backward(x_seq, u_seq)
        x_seq, u_seq = controller.forward(x_seq, u_seq, k_seq, kk_seq)
        # print(x_seq[-1])
        error_i = np.linalg.norm(x_seq[-1]-goal_state[:7])
        error.append(error_i)
    for t in range(time_horizon):
        state = env.step(u_seq[t])
        states.append(state)
        img = env.render()
        frames.append(img)

    # making a gif
    write_apng("./fig/iLQG Double Inverted Pendulum.gif", frames, delay=10)
    print("creating animated gif, please wait")
    states = x_seq
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
    plt.savefig('./fig/Cart_States_iLQG.png')
    plt.close()

    error = np.array(error)
    plt.plot(error)
    plt.title('Goal Error')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.savefig('./fig/Cart_Error_iLQG.png')
    return 0