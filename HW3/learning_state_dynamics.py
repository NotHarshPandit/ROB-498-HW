import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]

def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    # making a list of collected data
    collected_data = []
    action_space,state_space = 3,3
    # for each trajector in num of trajectories
    for trajectory in range (num_trajectories):
        # initializing a zero array of appropriate size for states and actions
        # print(env.observation_space)
        state_list = np.zeros((trajectory_length+1,state_space),dtype=np.float32)
        action_list = np.zeros((trajectory_length,action_space),dtype=np.float32)
        # for each timestep in trajectory length
        x_0 = env.reset()
        state_list[0,:] = x_0
        for t in range (trajectory_length):
            # sample a random action from env
            action = env.action_space.sample()
            # get the next state from the sampled action
            state,reward,done,info = env.step(action)
            # append the state and action list
            state_list[t+1,:] = state
            action_list[t,:] = action
            if done:
                env.reset()
        # make a state and action dictionary for every trajectory
        trajectory_dictionary = {
            "states":state_list,
            'actions':action_list
        }
        # append the dictionary for every trajectory
        collected_data.append(trajectory_dictionary)
    # ---
    return collected_data
def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:
    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    # Create a dataset from the single-step data
    dataset = SingleStepDynamicsDataset(collected_data)
    
    # Split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    # ---
    return train_loader, val_loader
def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)
    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:
    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    # create a dataset from collected data
    dataset = MultiStepDynamicsDataset(collected_data)
    # Split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    # ---
    return train_loader, val_loader
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
        trajectory_index = item // self.trajectory_length
        transition_index = item % self.trajectory_length
        full_states = self.data[trajectory_index]['states']
        full_actions = self.data[trajectory_index]['actions']
        sample['state'] = torch.tensor(full_states[transition_index],dtype=torch.float32)
        sample['action'] = torch.tensor(full_actions[transition_index], dtype=torch.float32)
        sample['next_state'] = torch.tensor(full_states[transition_index+1],dtype=torch.float32)
        # ---
        return sample
class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.
    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """
    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps
    def __len__(self):
        return len(self.data) * (self.trajectory_length)
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
            'next_state': None
        }
        # --- Your code here
        trajectory_index = item // self.trajectory_length
        transition_index = item % self.trajectory_length
        full_states = self.data[trajectory_index]['states']
        full_actions = self.data[trajectory_index]['actions']
        sample['state'] = torch.tensor(full_states[transition_index],dtype=torch.float32)
        sample['action'] = torch.tensor(full_actions[transition_index:transition_index+self.num_steps], dtype=torch.float32)
        sample['next_state'] = torch.tensor(full_states[transition_index+1:transition_index+1+self.num_steps],dtype=torch.float32)
        # ---
        return sample
class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.
    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}
    """
    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length
    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        # Calculate radius of gyration
        rg = ((self.l**2 + self.w**2)/12)**(1/2)
        x_pred, y_pred, theta_pred = pose_pred[:, 0], pose_pred[:, 1], pose_pred[:, 2]
        x_target, y_target, theta_target = pose_target[:, 0], pose_target[:, 1], pose_target[:, 2]
        se2_pose_loss = F.mse_loss(x_pred, x_target) + F.mse_loss(y_pred, y_target) + rg * F.mse_loss(theta_pred, theta_target)
        return se2_pose_loss
class SingleStepLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn
    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        pred_state = model(state,action)
        single_step_loss = self.loss(pred_state,target_state)
        # ---
        return single_step_loss
class MultiStepLoss(nn.Module):
    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount
    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        # initializing loss to zero
        batch_size, horizon, action_dim = actions.shape  # (500, 4, 3)
        # print("batch_size: ",batch_size)
        # print("horizon: ",horizon)
        # print("action_dim: ",action_dim)
        # print("target states shape: ",target_states.shape)
        multi_step_loss = 0  # Initialize loss
        next_state = state  # Initial state
        for i in range(horizon):  # Iterate over the horizon (4 timesteps)
            next_state = model(next_state, actions[:, i, :])  # Predict next state at step i
            target_state = target_states[:, i, :]  # Get corresponding target state
            step_loss = self.loss(next_state, target_state)  # Compute loss
            multi_step_loss += (self.discount ** i) * step_loss  # Apply discount and accumulate
        # ---
        return multi_step_loss
class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.hidden_feature_size = 100
        self.linear1 = torch.nn.Linear(self.state_dim + self.action_dim,self.hidden_feature_size)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.hidden_feature_size,self.hidden_feature_size)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(self.hidden_feature_size,self.state_dim)
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
        next_state = torch.cat((action,state),dim = -1 )
        next_state = self.linear1(next_state)
        next_state = self.activation1(next_state)
        next_state = self.linear2(next_state)
        next_state = self.activation2(next_state)
        next_state = self.linear3(next_state)
        # ---
        return next_state
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
        self.hidden_feature_size = 100
        self.linear1 = torch.nn.Linear(self.state_dim + self.action_dim,self.hidden_feature_size)
        self.activation1 = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(self.hidden_feature_size,self.hidden_feature_size)
        self.activation2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(self.hidden_feature_size,self.state_dim)
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
        next_state = torch.cat((state,action),dim = -1 )
        next_state = self.linear1(next_state)
        next_state = self.activation1(next_state)
        next_state = self.linear2(next_state)
        next_state = self.activation2(next_state)
        next_state = self.linear3(next_state) + state
        # ---
        return next_state
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
    # weight matrix for the cost
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=state.dtype)
    # calculating the cost
    state_diff = state - target_pose.unsqueeze(0)  # Ensure target_pose is broadcastable
    cost = torch.einsum('bi,ij,bj->b', state_diff, Q, state_diff)
    # ---
    return cost
def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    # obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    # obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    # box_size = BOX_SIZE  # scalar for parameter w
    # in_collision = None
    # # --- Your code here
    # in_collision = []
    # # storing the data of obstacle
    # obstacle_state = torch.tensor([*obstacle_centre, 0, *obstacle_dims])
    # # getting the number of iterations
    # B = state.shape[0]
    # # for each iteration
    # for i in range(B):
    # #   get the state for that iteration
    #   state_i = state[i]
    # #   storing the data of the box for that iteration
    #   box = torch.tensor([*state_i[:2],state_i[2] - np.pi/2,BOX_SIZE,BOX_SIZE])
    # #   check for collision
    #   in_collision.append(check_rectangle_collision(box, obstacle_state))
    # in_collision = torch.tensor(in_collision,dtype=torch.float32)
    # ---
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w

    # Extract x, y positions from state (assuming they are the first two dimensions)
    state_pos = state[:, :2]  # (B, 2)

    # Compute absolute difference between state positions and obstacle center
    diff = torch.abs(state_pos - obstacle_centre)  # (B, 2)

    # Compute collision check (AABB collision check)
    in_collision = (diff <= (obstacle_dims / 2 + box_size / 2)).all(dim=1).float()

    return in_collision
    return in_collision
def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    # print("Inside obstacle_avoidance_pushing_cost_function")
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    collisions = collision_detection(state).to(torch.float32)
    # print(collisions)
    # print("collisions size: ")
    # print(collisions.shape)
    # weight matrix for the cost function
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=state.dtype)
    # calculating the cost
    state_diff = state - target_pose.unsqueeze(0)  # Ensure target_pose is broadcastable
    # cost = torch.einsum('bi,ij,bj->b', state_diff, Q, state_diff)
    cost = (state_diff @ Q * state_diff).sum(dim=1)
    # print(cost)
    # print("cost size: ")
    # print(cost.shape)
    cost += 100*collisions
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
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.95 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.3
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)
    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        next_state = self.model(state,action)
        # ---
        return next_state
    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        print("Control 1")
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()
        # ---
        print("Control 2")
        action_tensor = self.mppi.command(state_tensor)

        # --- Your code here
        print("Control 3")
        action = action_tensor.squeeze(0).detach().numpy() 
        # ---
        return action
# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
def check_rectangle_collision(rectangle1, rectangle2):
    # getting the edges for the rectangle
    edges1 = get_rectangle_edges(rectangle1)
    edges2 = get_rectangle_edges(rectangle2)
    # checking for intersection between edges
    for edge1 in edges1:
        for edge2 in edges2:
            if check_line_intersection(edge1[0], edge1[1], edge2[0], edge2[1]):
                return True
    return False
    
def get_rectangle_edges(rectangle):
    edges = []
    vertices = rectangle_vertices(rectangle)
    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i + 1) % len(vertices)] 
        edges.append((start, end))
    return edges
def rectangle_vertices(rectangle):
    # center of the rectangle
    center = rectangle[:2].numpy()
    x,y = center
    # orientation of the rectangle
    theta = rectangle[2].numpy()
    # width and height of the rectangle
    width, height = rectangle[3:].numpy()
    # calculating the vertices based on center and width and height
    vertices = np.array([
        [x - width / 2, y - height / 2],
        [x + width / 2, y - height / 2],
        [x + width / 2, y + height / 2],
        [x - width / 2, y + height / 2]
    ])
    # rotation matrix for the orientation
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # taking the vertices in the center frame and then back to world frame
    vertices = np.dot(vertices - center[:2], rotation_matrix.T) + center[:2]
    return vertices
def check_line_intersection(p1, p2, p3, p4):
    # Check if two line segments intersect based on orientation of points
    p1 = np.array([p1[0], p1[1], 0])
    p2 = np.array([p2[0], p2[1], 0])
    p3 = np.array([p3[0], p3[1], 0])
    p4 = np.array([p4[0], p4[1], 0])
    # Calculate cross products and extract z-component
    d1 = np.cross(p4 - p3, p1 - p3)[2]
    d2 = np.cross(p4 - p3, p2 - p3)[2]
    d3 = np.cross(p2 - p1, p3 - p1)[2]
    d4 = np.cross(p2 - p1, p4 - p1)[2]
    if np.sign(d1) != np.sign(d2) and np.sign(d3) != np.sign(d4):
        return True
    # Check for collinear points (optional)
    if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
        # Additional check to see if any point lies on the other segment
        if (min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and
            min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1])):
            return True
        if (min(p1[0], p2[0]) <= p4[0] <= max(p1[0], p2[0]) and
            min(p1[1], p2[1]) <= p4[1] <= max(p1[1], p2[1])):
            return True
        if (min(p3[0], p4[0]) <= p1[0] <= max(p3[0], p4[0]) and
            min(p3[1], p4[1]) <= p1[1] <= max(p3[1], p4[1])):
            return True
        if (min(p3[0], p4[0]) <= p2[0] <= max(p3[0], p4[0]) and
            min(p3[1], p4[1]) <= p2[1] <= max(p3[1], p4[1])):
            return True
    return False
# ---
# ============================================================
