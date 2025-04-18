import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym


class CartpoleEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.cartpole = None
        super().__init__(*args, **kwargs)

    def step(self, control):
        """
            Steps the simulation one timestep, applying the given force
        Args:
            control: np.array of shape (1,) representing the force to apply

        Returns:
            next_state: np.array of shape (4,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (4,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        self.cartpole = p.loadURDF('cartpole.urdf')
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (4,) representing cartpole state [x, theta, x_dot, theta_dot]

        """

        x, x_dot = p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot = p.getJointState(self.cartpole, 1)[0:2]
        return np.array([x, theta, x_dot, theta_dot])

    def set_state(self, state):
        x, theta, x_dot, theta_dot = state
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta, targetVelocity=theta_dot)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=-30, high=30)  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
        theta_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]
        state_space = gym.spaces.Box(
            low=np.array([x_lims[0], theta_lims[0], x_dot_lims[0], theta_dot_lims[0]], dtype=np.float32),
            high=np.array([x_lims[1], theta_lims[1], x_dot_lims[1], theta_dot_lims[1]],
                          dtype=np.float32))  # linear force # TODO: Verify that they are correct
        return state_space

    def _setup_camera(self):
        self.render_h = 240
        self.render_w = 320
        base_pos = [0, 0, 0]
        cam_dist = 2
        cam_pitch = 0.3
        cam_yaw = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=self.render_w / self.render_h,
                                                        nearVal=0.1,
                                                        farVal=100.0)

    def linearize_numerical(self, state, control, eps=1e-3):
        """
            Linearizes cartpole dynamics around linearization point (state, control). Uses numerical differentiation
        Args:
            state: np.array of shape (4,) representing cartpole state
            control: np.array of shape (1,) representing the force to apply
            eps: Small change for computing numerical derivatives
        Returns:
            A: np.array of shape (4, 4) representing Jacobian df/dx for dynamics f
            B: np.array of shape (4, 1) representing Jacobian df/du for dynamics f
        """
        A, B = None, None
        # --- Your code here

        # Initialize A and B as 4x4 and 4x1 zero matrices
        A = np.zeros((4, 4))
        B = np.zeros((4, 1))
        # Compute the Jacobian using symmetric finite differences formula
        for i in range(4):
            delta_state = np.zeros_like(state)
            delta_state[i] = eps
            A[:, i] = (self.dynamics(state + delta_state, control) - self.dynamics(state - delta_state, control)) / (2 * eps)
        B = (self.dynamics(state, control + eps) - self.dynamics(state, control - eps)) / (2 * eps)
        B = np.reshape(B, (4, 1))
        # ---
        return A, B


def dynamics_analytic(state, control):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 4) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 4) representing the next cartpole state

    """
    next_state = None
    dt = 0.05
    g = 9.81
    mc = 1
    mp = 0.1
    l = 0.5

    # --- Your code here

    # split states into x, theta, xdot, thetadot
    x,theta,x_dot,theta_dot = torch.chunk(state, 4, dim=1)

    # finding theta_double_dot and x_double_dot
    theta_double_dot = (g*torch.sin(theta) - (torch.cos(theta)*((control + (mp*l*theta_dot*theta_dot*torch.sin(theta)))/(mc+mp))))/(l*((4/3)-(mp*torch.cos(theta)*torch.cos(theta)/(mc+mp))))
    x_double_dot = (control + (mp*l*(torch.sin(theta)*theta_dot**2 - (theta_double_dot*torch.cos(theta)))))/(mc+mp)
    
    # using the analytic equations to find t+1 state
    x_dot_t1 = x_dot + dt*x_double_dot
    theta_dot_t1 = theta_dot + dt*theta_double_dot
    x_t1 = x + dt*x_dot_t1
    theta_t1 = theta + dt*theta_dot_t1

    next_state = torch.column_stack((x_t1, theta_t1, x_dot_t1, theta_dot_t1))
    # ---

    return next_state


def linearize_pytorch(state, control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (4,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (4, 4) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (4, 1) representing Jacobian df/du for dynamics f

    """
    A, B = None, None
    # --- Your code here

    # setting required gradient to True for computing the gradient
    state.requires_grad_(True)
    control.requires_grad_(True)

    # reshaping the state and control tensors
    state = torch.reshape(state,(1,4))
    control = torch.reshape(control,(1,1))

    # computing the next state using the dynamics_analytic function
    next_state = dynamics_analytic(state, control)

    # initializing A as a 4x4 zero tensor
    A = torch.zeros((4, 4))
    for i in range(4):
        grad_A = torch.autograd.grad(next_state[0, i], state, retain_graph=True, create_graph=True)[0]
        A[i, :] = grad_A.squeeze(0)

    # initializing B as a 4x1 zero tensor
    B = torch.zeros((4, 1))
    for i in range(4):
        grad_B = torch.autograd.grad(next_state[0, i], control, retain_graph=True, create_graph=True)[0]
        B[i, :] = grad_B.squeeze(0)
    # ---
    return A.detach(), B.detach()
