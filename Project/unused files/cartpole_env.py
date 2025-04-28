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
            next_state: np.array of shape (6,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (6,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        self.cartpole = p.loadURDF('utils/double_pendulum_on_cart.urdf')
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 2, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (6,) representing cartpole state [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]

        """
        # prismatic joint
        x, x_dot = p.getJointState(self.cartpole, 0)[0:2] 
        # revolute joint 1
        theta1, theta1_dot = p.getJointState(self.cartpole, 1)[0:2]
        # revolute joint 2
        theta2, theta2_dot = p.getJointState(self.cartpole, 2)[0:2]
        return np.array([x, theta1,theta2-theta1, x_dot, theta1_dot,theta2_dot-theta1_dot])

    def set_state(self, state):
        x, theta1,theta2, x_dot, theta1_dot,theta2_dot = state
        theta2 = theta1 + theta2
        theta2_dot = theta1_dot + theta2_dot
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta1, targetVelocity=theta1_dot)
        p.resetJointState(self.cartpole, 2, targetValue=theta2, targetVelocity=theta2_dot)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=np.array([-30.0]), high=np.array([30.0]), dtype=np.float32)  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        """
        Defines the state space of the environment.
        Returns:
            gym.spaces.Box: Bounds on [x, theta1, theta2, x_dot, theta1_dot, theta2_dot].
        """
        x_lims = [-5, 5]  # Cart position
        theta_lims = [-np.pi, np.pi]  # Pole angles
        x_dot_lims = [-10, 10]  # Cart velocity
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]  # Pole angular velocities

        low = np.array([
            x_lims[0],
            theta_lims[0],
            theta_lims[0],
            x_dot_lims[0],
            theta_dot_lims[0],
            theta_dot_lims[0]
        ], dtype=np.float32)

        high = np.array([
            x_lims[1],
            theta_lims[1],
            theta_lims[1],
            x_dot_lims[1],
            theta_dot_lims[1],
            theta_dot_lims[1]
        ], dtype=np.float32)

        return gym.spaces.Box(low=low, high=high, dtype=np.float32)


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
    
def normalize_angle(angle):
    normalized_angle = angle % (2 * np.pi)  # Wrap the angle to the range [0, 2*pi)
    
    if normalized_angle > np.pi:
        normalized_angle -= 2 * np.pi  # Map values greater than pi to the negative range
    
    return normalized_angle


def dynamics_analytic(state, control):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 6) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 6) representing the next cartpole state

    """
    next_state = None
    dt = 0.05
    g = 9.81
    m0 = 1.0  # mass of cart
    m1 = 1.0  # mass of pole 1
    m2 = 1.0  # mass of pole 2
    L1 = 1 #length of pole 1
    L2 = 1  #length of pole 1

    # total mass of the system
    d1 = m0 + m1 + m2
    d2 = L1*((0.5*m1) + m2)
    d3 = 0.5*m2*L2
    d4 = (m2 + (m1/3))*(L1**2)
    d5 = 0.5*m2*L1*L2
    d6 = (m2*(L2**2))/3

    f1 = ((m1*0.5) + m2)*g*L1
    f2 = 0.5*m2*g*L2

    # split states into x, theta1, theta2, x_dot, theta_dot, theta1_dot, theta2_dot
    x,theta1,theta2,x_dot,theta1_dot,theta2_dot = torch.chunk(state, 6, dim=1)

    # D and inverse of D
    D = torch.tensor([[d1, d2*torch.cos(theta1) , d3*torch.cos(theta2)], 
                  [d2*torch.cos(theta1), d4, d5*torch.cos(theta1 - theta2)],
                  [d3*torch.cos(theta2), d5*torch.cos(theta1 - theta2), d6]],dtype=x.dtype)
    D_inv = torch.linalg.inv(D)
    D_inv = D_inv.clone().detach().to(dtype=x.dtype)


    C = torch.tensor([[0, -d2*torch.sin(theta1)*theta1_dot, -d3*torch.sin(theta2)*theta2_dot],
                  [d2*torch.sin(theta1)*theta1_dot, 0, d5*torch.sin(theta1 - theta2)*theta2_dot],
                  [d3*torch.sin(theta2)*theta2_dot, -d5*torch.sin(theta1 - theta2)*theta1_dot, 0]],dtype=x.dtype)
    
    G = torch.tensor([[0],
                  [-f1*torch.sin(theta1)],
                  [-f2*torch.sin(theta2)]],dtype=x.dtype)
    
    H = torch.tensor([[1],
                  [0],
                  [0]],dtype=x.dtype)
    
    control = control.to(dtype=x.dtype)

    # derivative = torch.matmul(x_coeff, torch.column_stack((x,theta1,theta2,x_dot,theta1_dot,theta2_dot))) + constant + action_coeff * control
    # using the analytic equations to find t+1 state
    x_double_dot, theta1_double_dot, theta2_double_dot = torch.matmul(D_inv, ((H*control) - (C @ torch.column_stack((x_dot, theta1_dot, theta2_dot)).T) - G))
    x_dot_t1 = x_dot + dt * x_double_dot
    theta1_dot_t1 = theta1_dot + dt * theta1_double_dot
    theta2_dot_t1 = theta2_dot + dt * theta2_double_dot

    x_t1 = x + dt * x_dot_t1
    theta_t1 = theta1 + dt * theta1_dot_t1
    theta2_t1 = theta2 + dt * theta2_dot_t1
    # stacking the next state
    next_state = torch.column_stack((x_t1, theta_t1, theta2_t1, x_dot_t1, theta1_dot_t1, theta2_dot_t1))

    # Normalize angles to be between -pi and pi
    # next_state[:,1] = normalize_angle(next_state[:,1])
    # next_state[:,2] = normalize_angle(next_state[:,2])  

    x_lims = [-5, 5]
    theta_lims = [-np.pi, np.pi]
    x_dot_lims = [-10, 10]
    theta_dot_lims = [-5 * np.pi, 5 * np.pi]

    # Convert to tensors for clamping
    low = torch.tensor([
        x_lims[0],
        theta_lims[0],
        theta_lims[0],
        x_dot_lims[0],
        theta_dot_lims[0],
        theta_dot_lims[0]
    ], dtype=next_state.dtype, device=next_state.device)

    high = torch.tensor([
        x_lims[1],
        theta_lims[1],
        theta_lims[1],
        x_dot_lims[1],
        theta_dot_lims[1],
        theta_dot_lims[1]
    ], dtype=next_state.dtype, device=next_state.device)

    # Clip the state within bounds
    # next_state = torch.clamp(next_state, min=low, max=high)

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
