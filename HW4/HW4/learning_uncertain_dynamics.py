import torch
import gpytorch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch import nn
from torch.distributions import MultivariateNormal
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, DISK_RADIUS, OBSTACLE_RADIUS, OBSTACLE_CENTRE
from utils import batch_cov

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)[:2]
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)[:2]


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here

        # ---
        return sample


class MultitaskGPModel(gpytorch.models.ExactGP):
    """
        Multi-task GP model for dynamics x_{t+1} = f(x_t, u_t)
        Each output dimension of x_{t+1} is represented with a seperate GP
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        # --- Your code here

        # ---

    def forward(self, x):
        """

        Args:
            x: torch.tensor of shape (B, dx + du) concatenated state and action

        Returns: gpytorch.distributions.MultitaskMultivariateNormal - Gaussian prediction for next state

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    def grad_mu(self, x):
        """
        Compute the gradient of the mean function
        Args:
            x: torch.tensor of shape (B, dx + du) concatenated state and action

        Returns:
            grad_mu: torch.tensor of shape (B, dx, dx + du) torch.tensor which is the Jacobian of the mean function

        """

        flatten = False
        if len(x.shape) < 2:
            M = 1
            x = x.reshape(M, -1)
            flatten = True

        # Get GP train x and y
        X = self.train_inputs[0]
        y = self.train_targets
        # N is datset size, M is query size
        N = X.shape[0]
        M = x.shape[0]

        # Compute difference
        diff = x.unsqueeze(1) - X.unsqueeze(0)  # M x N x d difference
        lengthscale = self.covar_module.base_kernel.lengthscale  # 2 x d
        W = 1.0 / (lengthscale ** 2)

        # Compute exponential term
        sq_diff = torch.sum(diff.unsqueeze(0) ** 2 * W.reshape(2, 1, 1, -1), dim=-1)  # 2 x M x N
        exponential_term = torch.exp(-0.5 * sq_diff)  # 2 x M x N

        # Compute gradient of Kernel
        sigma_f_sq = self.covar_module.outputscale

        # grad should be 2 x M x N x d
        grad_K = -W.reshape(2, 1, 1, -1) * diff.reshape(1, M, N, -1) * \
                 sigma_f_sq.reshape(2, 1, 1, 1) * exponential_term.unsqueeze(-1)

        # Compute gradient of mean function
        K = self.covar_module(X).evaluate()
        sigma_n_sq = self.likelihood.noise
        eye = torch.eye(N, device=x.device).reshape(1, N, N).repeat(2, 1, 1)
        mu = torch.linalg.solve(K + sigma_n_sq * eye, y.permute(1, 0).unsqueeze(-1))
        grad_mu = (grad_K.permute(0, 1, 3, 2) @ mu.reshape(2, 1, N, -1)).reshape(2, M, -1)

        if flatten:
            return grad_mu.reshape(2, -1)

        return grad_mu.permute(1, 0, 2)  # return shape M x 2 x 5


class ResidualMean(gpytorch.means.Mean):
    def __init__(self, batch_shape):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, input):
        """
        Residual mean function
        Args:
            input: torch.tensor of shape (N, dx + du) containing state and control vectors

        Returns:
            mean: torch.tensor of shape (dx, N) containing state vectors

        """
        mean = None
        # --- Your code here

        # ---
        return mean


class PushingDynamics(nn.Module):
    def __init__(self, propagation_method):
        super().__init__()
        self.propagation_method = propagation_method

    def forward(self, state, action):
        raise NotImplementedError

    def predict(self, state, action):
        raise NotImplementedError

    def propagate_uncertainty(self, mu, sigma, action):
        if self.propagation_method == 'certainty_equivalence':
            return self.propagate_uncertainty_certainty_equivalence(mu, sigma, action)
        if self.propagation_method == 'linearization':
            return self.propagate_uncertainty_linearization(mu, sigma, action)
        if self.propagation_method == 'moment_matching':
            return self.propagate_uncertainty_moment_matching(mu, sigma, action)

        raise ValueError('invalid self.propagation_method')

    def propagate_uncertainty_linearization(self, mu, sigma, action):
        raise NotImplementedError

    def propagate_uncertainty_moment_matching(self, mu, sigma, action, K=50):
        """
        Propagate uncertainty via moment matching with samples
        Args:
            mu: torch.tensor of shape (N, dx) consisting of mean of current state distribution
            sigma: torch.tensor of shape (N, dx, dx) covariance matrix of current state distribution
            action: torch.tensor of shape (N, du) action

        Returns:
            pred_mu: torch.tensor of shape (N, dx) consisting of mean of predicted state distribution
            pred_sigma: torch.tensor of shape (N, dx, dx) consisting of covariance matrix of predicted state distribution

        """

        pred_mu, pred_sigma = None, None

        # --- Your code here

        # ---

        return pred_mu, pred_sigma

    def propagate_uncertainty_certainty_equivalence(self, mu, sigma, action):
        """
            Propagate uncertainty via certainty equivalence, i.e. use the mean
        Args:
            mu: torch.tensor of shape (N, dx) consisting of mean of current state distribution
            sigma: torch.tensor of shape (N, dx, dx) covariance matrix of current state distribution
            action: torch.tensor of shape (N, du) action
        Returns:
            pred_mu: torch.tensor of shape (N, dx) consisting of mean of predicted state distribution
            pred_sigma: torch.tensor of shape (N, dx, dx) consisting of covariance matrix of predicted state distribution
        """
        pred_mu, pred_sigma = None, None
        # --- Your code here

        # ---
        return pred_mu, pred_sigma


class PushingDynamicsGP(PushingDynamics):

    def __init__(self, train_states, train_actions, train_next_states, likelihood,
                 propagation_method='certainty_equivalence'
                 ):
        super().__init__(propagation_method)
        self.gp_model = MultitaskGPModel(
            torch.cat((train_states, train_actions), dim=1),
            train_next_states,
            likelihood
        )

        self.likelihood = likelihood

    def forward(self, state, action):
        """
            Forward function for pushing dynamics
            This is the function that should be used when you are training your GP
        Args:
            state: torch.tensor of shape (N, dx)
            action: torch.tensor of shape (N, du)

        Returns:
            Prediction as a MultitaskMultivariateNormalDistribution, i.e. the result of calling self.gp_model

        """
        pred = None
        # --- Your code here

        # ---
        return pred

    def predict(self, state, action):
        """
            This is the method for predicting at test time

            This function includes the uncertainty from the likelihood, and also ensures that predictions
            are independent from one another.
        Args:
            state: torch.tensor of shape (N, dx)
            action: torch.tensor of shape (N, du)

        Returns:
            next_state_mu: torch.tensor of shape (N, dx)
            next_state_sigma: torch.tensor of shape (N, dx, dx)

        """

        # Get Gaussian prediction including likelihood uncertainty
        pred = self.likelihood(self.forward(state, action))

        # Get mean
        next_state_mu = pred.mean

        # Get covariance
        # Use stddev to ensure independence
        next_state_sigma = torch.diag_embed(pred.stddev ** 2)

        return next_state_mu, next_state_sigma

    def propagate_uncertainty_linearization(self, mu, sigma, action):
        """
            Propagate uncertainty via linearization
        Args:
            mu: torch.tensor of shape (N, dx) consisting of mean of current state distribution
            sigma: torch.tensor of shape (N, dx, dx) covariance matrix of current state distribution
            action: torch.tensor of shape (N, du) action

        Returns:
            pred_mu: torch.tensor of shape (N, dx) consisting of mean of predicted state distribution
            pred_sigma: torch.tensor of shape (N, dx, dx) consisting of covariance matrix of predicted state distribution

        """

        pred_mu, pred_sigma = None, None
        A = self.gp_model.grad_mu(torch.cat((mu, action), dim=-1))[:, :, :2]  # want B x 2 x 2

        # --- Your code here

        # ---
        return pred_mu, pred_sigma


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here

        # ---
        return next_state


class DynamicsNNEnsemble(PushingDynamics):

    def __init__(self, state_dim, action_dim, num_ensembles, propagation_method='moment_matching'):
        assert propagation_method in ['moment_matching', 'certainty_equivalence']
        super().__init__(propagation_method)
        self.models = nn.ModuleList([ResidualDynamicsModel(state_dim, action_dim) for _ in range(num_ensembles)])
        self.propagation_method = propagation_method

    def forward(self, state, action):
        """
            Forward function for dynamics ensemble
            You should use this during training
        Args:
            state: torch.tensor of shape (B, dx)
            action: torch.tensor of shape (B, du)

        Returns:
            Predicted next state for each of the ensembles
            next_state: torch.tensor of shape (B, N, dx) where N is the number of models in the ensemble

        """
        next_state = None
        # --- Your code here

        # ---
        return next_state

    def predict(self, state, action):
        """
            Predict function for NN ensemble
            You should use this during evaluation
            This will return the mean and covariance of the ensemble output
         Args:
            state: torch.tensor of shape (B, dx)
            action: torch.tensor of shape (B, du)

        Returns:
            Predicted next state for each of the ensembles
            pred_mu : torch.tensor of shape (B, dx)
            pred_sigma: torch.tensor of shape (B, dx, dx) covariance matrix

        """
        pred_mu, pred_sigma = None, None
        # --- Your code here

        # ---
        return pred_mu, pred_sigma


def train_dynamics_gp_hyperparams(model, likelihood, train_states, train_actions, train_next_states, lr):
    """
        Function which optimizes the GP Kernel & likelihood hyperparameters
    Args:
        model: gpytorch.model.ExactGP model
        likelihood: gpytorch likelihood
        train_states: (N, dx) torch.tensor of training states
        train_actions: (N, du) torch.tensor of training actions
        train_next_states: (N, dx) torch.tensor of training targets
        lr: Learning rate

    """
    # --- Your code here

    # ---


def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here

    # ---
    return cost


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.

    :param state: torch tensor of shape (B, dx + dx^2) First dx consists of state mean,
                  the rest is the state covariance matrix
    :param action: torch tensor of shape (B, du)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    obs_centre = torch.from_numpy(OBSTACLE_CENTRE).to(dtype=torch.float32)
    obs_radius = OBSTACLE_RADIUS
    disk_radius = DISK_RADIUS
    cost = None
    # --- Your code here

    # ---
    return cost


def obstacle_avoidance_pushing_cost_function_samples(state, action, K=10):
    """
    Compute the state cost for MPPI on a setup with obstacles, using samples to evaluate the expected collision cost

    :param state: torch tensor of shape (B, dx + dx^2) First dx consists of state mean,
                  the rest is the state covariance matrix
    :param action: torch tensor of shape (B, du)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    obs_centre = torch.from_numpy(OBSTACLE_CENTRE).to(dtype=torch.float32)
    obs_radius = OBSTACLE_RADIUS
    disk_radius = DISK_RADIUS
    cost = None
    # --- Your code here

    # ---
    return cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low) + 1e-5
        u_max = torch.from_numpy(env.action_space.high) - 1e-5
        noise_sigma = 0.25 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.001
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim + state_dim * state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size + state_size**2)
                      consisting of state mean and flattened covariance
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size + state_size**2) containing the predicted states mean
                 and covariance
        """
        next_state = None
        # --- Your code here

        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Recall that our current state is (state_size,), but we have set up our dynamics to work with state means and
           covariances. You need to use the current state to initialize a (mu, sigma) tensor. Given that we know the
           current state, the initial sigma should be zero.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here

        # ---
        return action
