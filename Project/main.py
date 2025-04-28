# importing all the libraries
from load_env import *
from check_analytics import check_anayltics
from iLQG_perfomance import iLQG_cartpole
from mppi_performance import mppi_cartpole
from ddp_perfomance import ddp_cartpole
import numpy as np

import builtins

def main():
    env = MyCartpoleEnv()
    start_state = np.array([0.1, 0.1,  0.1, 0, 0, 0], dtype=np.float32) # 6D state vector (x, theta1, theta2, x_dot, theta1_dot, theta2_dot)
    env.reset(start_state)
    # for plotting the analytical equations vs pybullet
    # check_anayltics(env)
    env.reset(start_state)
    # for plotting ILQG
    # iLQG_cartpole(env)
    ddp_cartpole(env)
    env.reset(start_state)
    # for plotting the mppi performance
    mppi_cartpole(env)
    return 0

if __name__ == "__main__":
    main()

