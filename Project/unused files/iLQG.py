import torch
from torch.distributions.multivariate_normal import MultivariateNormal
def get_cartpole_iLQG_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the cartpole environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 1
    state_size = 6
    hyperparams = {
        'lambda': None,
        'Q': None,
        'noise_sigma': None,
    }
    # --- Your code here
    hyperparams['lambda'] = 0.01
    hyperparams['Q'] = torch.tensor([[5, 0, 0, 0,0,0],
                        [0, 5, 0, 0,0,0],
                        [0,0, 5,0,0,0],
                        [0,0,0, 0.1,0,0],
                        [0,0,0, 0,0.1,0],
                        [0,0,0, 0.0,0,0.1]])
    hyperparams['noise_sigma'] = torch.eye(action_size) * 20
    # ---
    return hyperparams


class ILQG():
    """
    Class for the Iterative Linear Quadratic Gaussian (ILQG) controller.
    """
    def __init__(self, env, num_samples, horizon, hyperparams=None):
        """
        Initialize the ILQG controller.

        Args:
        param env: Simulation environment. Must have an action_space and a state_space.
        param num_samples: <int> Number of perturbed trajectories to sample
        param horizon: <int> Number of control steps into the future
        param hyperparams: <dic> containing the MPPI hyperparameters
        """
        self.env = env
        self.T = horizon
        self.K = num_samples
        self.lambda_ = hyperparams['lambda']
        self.action_size = env.action_space.shape[-1]
        self.state_size = env.state_space.shape[-1]
        self.goal_state = torch.zeros(self.state_size)  # This is just a container for later use
        self.Q = hyperparams['Q'] # Quadratic Cost Matrix (state_size, state_size)
        self.noise_mu = torch.zeros(self.action_size)
        self.noise_sigma = hyperparams['noise_sigma']  # Noise Covariance matrix shape (action_size, action_size)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = torch.zeros(self.action_size)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
    
    def reset(self):
        """
        Resets the nominal action sequence
        :return:
        """
        self.U = torch.zeros((self.T, self.action_size))# nominal action sequence (T, action_size)

    def command(self, state,action):
        """
        Run a iLQG step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return:
        """

        # action is (T,action_size)
        actions = self.U
        nominal_states = self._rollout_dynamics(state, actions)

        A_list, B_list = [], []
        for t in range(self.T):
            A_t, B_t = self.env.linearize_pytorch(nominal_states[t,:], self.U[t])
            A_list.append(A_t)
            B_list.append(B_t)
        perturbations = self.noise_dist.sample((self.T,1))    # shape (T, action_size)
        cost = self._compute_trajectory_cost(nominal_states, perturbations)
        print("cost shape", cost.shape)
        delta_u, K = self.backward_pass(A_list, B_list, nominal_states, self.U, cost)

       
    
        # A_t, B_t = self.env.linearize_pytorch(state, action)
        return action
    
    
    def backward_pass(self, A_list, B_list, nominal_states, perturbations, cost):
        """
        Perform the backward pass of iLQG to compute the optimal control sequence for a single trajectory.
        Args:
            A_list: List of Jacobian matrices for state dynamics at each time step.
            B_list: List of Jacobian matrices for action dynamics at each time step.
            nominal_states: Nominal states for the trajectory.
            perturbations: Perturbations to be applied to nominal control actions.
            cost: Total cost for the trajectory.

        Returns:
            delta_u: Perturbations to be applied at each time step.
            K: Feedback gain at each time step.
        """
        # Initialize variables for the backward pass
        P = torch.zeros(self.state_size, self.state_size)  # cost-to-go matrix for the last time step
        K = torch.zeros(self.T, self.state_size, self.action_size)  # Feedback gain at each time step
        delta_u = torch.zeros(self.T, self.action_size)  # Perturbation at each time step
        
        # Initialize the final cost-to-go for the last time step (assuming final state cost is given)
        P = self.Q @ (nominal_states[-1] - self.goal_state)  # State cost at final time step
        
        # Backward pass from the final time step (T-1) to 0
        for t in reversed(range(self.T)):
            # Compute the state cost-to-go at time step t
            state_cost = cost[t]  # Scalar cost for the current time step
            P_t = state_cost + A_list[t].T @ P @ A_list[t]  # Updated cost-to-go based on linearized dynamics
            
            # Compute the feedback gain K_t
            Q_t = A_list[t].T @ P_t @ A_list[t] + B_list[t].T @ P_t @ B_list[t] + self.R  # Regularized term
            K_t = -torch.inverse(Q_t) @ A_list[t].T @ P_t @ B_list[t]  # Feedback gain matrix
            
            # Store the feedback gain for this time step
            K[t] = K_t
            
            # Compute the perturbation delta_u_t at time step t
            delta_u_t = K_t @ (nominal_states[t] - self.goal_state)
            delta_u[t] = delta_u_t
            
            # Update the cost-to-go for the previous time step
            P = P_t
        return delta_u, K

    def _rollout_dynamics(self, state_0, actions):
        """
        Roll out the environment dynamics from state_0 and taking the control actions given by actions
        :param state_0: torch tensor of shape (state_size,)
        :param actions: torch tensor of shape (T, action_size)
        :return:
         * trajectory: torch tensor of shape (T, state_size) containing the states along the trajectories given by
                       starting at state_0 and taking actions.
                       This tensor contains K trajectories of T length.
         TIP 1: You may need to call the self._dynamics method.
         TIP 2: At most you need only 1 for loop.
        """
        state = state_0
        trajectory = None
        # --- Your code here
        trajectory = torch.zeros((self.T, self.state_size), dtype=state_0.dtype)
        for t in range(self.T):
            # print(actions[t, :])
            state = self.env.dynamics_analytic(state, actions[t, :])  # Compute next state
            trajectory[t, :] = state  # Store current state in trajectory
        # ---
        return trajectory
    
    def _compute_trajectory_cost(self, trajectory, perturbations):
        """
        Compute the costs for a single trajectory.
        :param trajectory: torch tensor of shape (T, state_size) containing the states along the trajectory
        :param perturbations: torch tensor of shape (T, action_size) containing perturbations at each time step
        :return:
        - total_trajectory_cost: torch tensor of shape (1,) containing the total trajectory cost for the single trajectory
        """
        total_trajectory_cost = 0.0  # Initialize total cost

        # Compute state costs (quadratic state cost)
        state_cost = 0.0
        for t in range(self.T):
            state_diff = trajectory[t, :] - self.goal_state  # State difference
            state_cost += torch.matmul(state_diff, torch.matmul(self.Q, state_diff))  # (state - goal)^T Q (state - goal)

        # Compute action costs
        action_cost = 0.0
        for t in range(self.T):
            nominal_action = self.U[t, :]  # Nominal action (without perturbation)
            perturbation = perturbations[t, :]  # Perturbation at time step t
            action_cost += torch.matmul(nominal_action, torch.matmul(self.noise_sigma_inv, perturbation))  # u^T Sigma_inv perturbation

        # Compute total trajectory cost
        total_trajectory_cost = state_cost + action_cost
        return total_trajectory_cost

    def _nominal_trajectory_update(self, trajectory_costs, perturbations):
        """
        Update the nominal action sequence (self.U) given the trajectory costs and perturbations
        :param trajectory_costs: torch tensor of shape (K,)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return: No return, you just need to update self.U

        TIP: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references about the action update.
        """
        # --- Your code here
        beta = torch.min(trajectory_costs)
        gamma = torch.exp(-(trajectory_costs - beta)/self.lambda_)
        eeta = torch.sum(gamma)
        weights = gamma / eeta
        # Ensure proper broadcasting (reshape weights to (K, 1, 1))
        weights = weights.view(-1, 1, 1)

        # Compute the weighted sum of perturbations
        weighted_perturbations = weights * perturbations
        self.U += torch.sum(weighted_perturbations, dim=0)  # Update nominal actions
        # ---

    def _dynamics(self, state, action):
        """
        Query the environment dynamics to obtain the next_state in a batched format.
        :param state: torch tensor of size (...., state_size)
        :param action: torch tensor of size (..., action_size)
        :return: next_state: torch tensor of size (..., state_size)
        """
        next_state = self.env.batched_dynamics(state.cpu().detach().numpy(), action.cpu().detach().numpy())
        next_state = torch.tensor(next_state, dtype=state.dtype)
        return next_state

