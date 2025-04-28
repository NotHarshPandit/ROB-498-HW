# importing all the libraries
from load_env import *
from check_analytics import check_anayltics
# from iLQG_perfomance import iLQG_cartpole
from mppi_performance import mppi_cartpole
from ddp_performance import ddp_cartpole
import numpy as np

def main():
    env = MyCartpoleEnv()
    start_state = np.array([0.1, 0.1,  0.1, 0, 0, 0], dtype=np.float32) # 6D state vector (x, theta1, theta2, x_dot, theta1_dot, theta2_dot)

    # for plotting the DDP performance
    env.reset(start_state)
    ddp_cartpole(env)
    
    # for plotting the mppi performance
    env.reset(start_state)
    mppi_cartpole(env)
    return 0

if __name__ == "__main__":
    main()

