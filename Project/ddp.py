import gym
from autograd import grad, jacobian
import autograd.numpy as np
from cartpole_env import MyCartpoleEnv

import matplotlib.pyplot as plt



def dynamics_analytic_np(state, action):
    """
        Computes next state of cartpole given current state and action using analytic dynamics

    Args:
        state: numpy array of shape (6,) representing cartpole state
        action: numpy array of shape (1,) representing the force to apply

    Returns:
        next_state: numpy array of shape (6,) representing next cartpole state
    """
    m0 = 1.0
    m1 = 0.1
    m2 = 0.1
    L1 = 1.0
    L2 = 1.0
    g = 9.81  # gravity

    action = -action  # Negative action to match the dynamics equations (control input flip)
    
    # Extract individual states from the state_0 array
    x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
    
    # Construct D, C, G, H matrices (dynamics model)
    # total mass of the system
    d1 = m0 + m1 + m2
    d2 = L1*((0.5*m1) + m2)
    d3 = 0.5*m2*L2
    d4 = (m2 + (m1/3))*(L1**2)
    d5 = 0.5*m2*L1*L2
    d6 = (m2*(L2**2))/3

    f1 = ((m1*0.5) + m2)*g*L1
    f2 = 0.5*m2*g*L2

    
    D = np.array([[d1, d2*np.cos(theta1), d3*np.cos(theta2)], 
                [d2*np.cos(theta1), d4, d5*np.cos(theta1 - theta2)],
                [d3*np.cos(theta2), d5*np.cos(theta1 - theta2), d6]])

    C = np.array([[0, -d2*np.sin(theta1)*theta1_dot, -d3*np.sin(theta2)*theta2_dot],
                [d2*np.sin(theta1)*theta1_dot, 0, d5*np.sin(theta1 - theta2)*theta2_dot],
                [d3*np.sin(theta2)*theta2_dot, -d5*np.sin(theta1 - theta2)*theta1_dot, 0]])

    G = np.array([[0],
                [-f1*np.sin(theta1)],
                [-f2*np.sin(theta2)]])
    
    H = np.array([[1],
                [0],
                [0]])

    # Compute the second derivatives of the state variables (x_ddot, theta_1_ddot, theta_2_ddot)
    state_dot = np.array([[x_dot], [theta1_dot], [theta2_dot]])  # Shape (3, 1)
    temp = -np.linalg.inv(D) @ (C @ state_dot + G + H * action)  # Shape (3, 1)
    
    # Extract the accelerations (second derivatives of the states)
    x_dot_dot = temp[0, 0]
    theta_1_dot_dot = temp[1, 0]
    theta_2_dot_dot = temp[2, 0]
    
    # Compute the new state using the previous state and computed accelerations
    dt = 0.05  # time step, adjust as necessary
    x_dot_new = x_dot + dt * x_dot_dot
    theta_1_dot_new = theta1_dot + dt * theta_1_dot_dot
    theta_2_dot_new = theta2_dot + dt * theta_2_dot_dot
    x_new = x + dt * x_dot_new
    theta_1_new = theta1 + dt * theta_1_dot_new
    theta_2_new = theta2 + dt * theta_2_dot_new

    # Pack the new state
    next_state = np.array([x_new, theta_1_new, theta_2_new, x_dot_new, theta_1_dot_new, theta_2_dot_new])
    lims = [
    [-5, 5],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-10, 10],
    [-5 * np.pi, 5 * np.pi],
    [-5 * np.pi, 5 * np.pi]]
    next_state = clip_state(next_state, lims)
    return next_state

def clip_state(state, lims):
    clipped = []
    for i in range(len(state)):
        clipped.append(np.clip(state[i], lims[i][0], lims[i][1]))
    return np.stack(clipped, axis=0)

class DDPController:
    def __init__(self,
                 umax, state_dim, pred_time=50):
        self.pred_time = pred_time
        self.umax = umax
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = dynamics_analytic_np
        self.lf = self.final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.l_x = grad(self.running_cost, 0)
        self.l_u = grad(self.running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)
        self.mu = 0.1
        self.alpha = 1

    def running_cost(self,state, action):
        cost = None
        Q = np.diag((5,5,5,0.1,0.1,0.1))
        # Q = np.diag((0,0,0,0,0,0))
        R = np.array([[0.5]])
        goal_state = np.array([0, 0, 0, 0, 0, 0])
        state_diff = state - goal_state
        cost = np.dot(state_diff.T, np.dot(Q, state_diff)) + np.dot(action.T, np.dot(R, action))
        return cost
    
    def final_cost(self,state):
        cost = None
        Q = np.diag((5,5,5,0.1,0.1,0.1))
        goal_state = np.array([0, 0, 0, 0, 0, 0])
        state_diff = state - goal_state
        cost = np.dot(state_diff.T, np.dot(Q, state_diff))
        return cost

    def backward(self, x_seq, u_seq):
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        K_seq = []
        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
              np.matmul(np.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t)
            tmp = np.matmul(f_u_t.T, self.v_xx[t + 1])
            # q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t)
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_uu(x_seq[t], u_seq[t])))
            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.matmul(tmp, f_x_t) +\
             np.dot(self.v_x[t + 1], np.squeeze(self.f_ux(x_seq[t], u_seq[t])))
            
            temp = np.matmul(f_u_t.T,(self.v_xx[t + 1] + self.mu*np.eye(self.v_xx[t + 1].shape[0])))
            # q_uu_hat = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(temp, f_u_t) + \
            # np.dot(self.v_x[t + 1], np.squeeze(self.f_uu(x_seq[t], u_seq[t])))
        
            # q_ux_hat = self.l_ux(x_seq[t], u_seq[t]) + np.matmul(temp, f_x_t) +\
                #    np.dot(self.v_x[t+1],np.squeeze(self.f_ux(x_seq[t], u_seq[t])))
            # print("quu shape", q_uu.shape)
            # print("quu hat shape",q_uu_hat.shape)
            # inv_q_uu_hat = np.linalg.inv(q_uu)
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u)
            K = -np.matmul(inv_q_uu, q_ux)
            delta_v = 0.5 * np.matmul(q_u, k)
            self.v[t] += delta_v
            self.v_x[t] =  q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + np.matmul(q_ux.T, K)
            k_seq.append(k)
            K_seq.append(K)
        k_seq.reverse()
        K_seq.reverse()
        return k_seq, K_seq

    def forward(self, x_seq, u_seq, k_seq, K_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = self.alpha*k_seq[t] + np.matmul(K_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        x_lims = [-5, 5] 
        theta_lims = [-np.pi, np.pi]
        theta_1_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]
        theta_dot_1_lims = [-5 * np.pi, 5 * np.pi]
        x_seq_hat[0] = np.clip(x_seq_hat[0], x_lims[0], x_lims[1])
        x_seq_hat[1] = np.clip(x_seq_hat[1], theta_lims[0], theta_lims[1])
        x_seq_hat[2] = np.clip(x_seq_hat[2], theta_1_lims[0], theta_1_lims[1])
        x_seq_hat[3] = np.clip(x_seq_hat[3], x_dot_lims[0], x_dot_lims[1])
        x_seq_hat[4] = np.clip(x_seq_hat[4], theta_dot_lims[0], theta_dot_lims[1])
        x_seq_hat[5] = np.clip(x_seq_hat[5], theta_dot_1_lims[0], theta_dot_1_lims[1])
        u_seq_hat = np.clip(u_seq_hat, -self.umax, self.umax)
        return x_seq_hat, u_seq_hat
