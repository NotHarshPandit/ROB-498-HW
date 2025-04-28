from ddp import *
# from ddp_utils import *
import autograd.numpy as np
from autograd import grad, jacobian

class DDPController(object):
    def __init__(self, start_state, goal_state, hyperparams):
        self.start_state = start_state
        self.curr_state = start_state
        self.goal_state = goal_state
        self.state_dim = 6
        self.action_dim = 1
        self.counter = 0
        
        self.error_Q = hyperparams['error_Q']
        max_iters = hyperparams['max_iters']
        epsilon = hyperparams['epsilon']
        horizon = hyperparams['horizon']
        backtrack_max_iters = hyperparams['backtrack_max_iters']
        decay = hyperparams['decay']
    
        
        self.dynamics = self.dynamics_analytic_np           # x_{t+1} = f(x_t, u_t): analytic dynamics function
        self.compute_cost = self.compute_cost   # J: cost function
        self.running_cost = self.running_cost   # g/L: running cost function
        self.terminal_cost = self.terminal_cost # h/phi: terminal cost function
        self.max_iters = max_iters         # maximal number of iterations
        self.epsilon = epsilon             # tolerance of cost
        self.T = horizon                   # length of state (0,1,...,T) action length (0,1,...,T-1)
        self.B = backtrack_max_iters       # maxmimal number of backtracking iterations
        self.decay = decay                 # decay: decay coefficients
        self.state_dim = 6         # dimension of state
        self.action_dim = 1      # dimension of action

        # setting the dynamics and its derivatives
        self.f = self.dynamics
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_xu = jacobian(self.f_x, 1)
        self.f_ux = jacobian(self.f_u, 0)
        self.f_uu = jacobian(self.f_u, 1)        
        
        # setting the running cost and its derivatives
        self.lx = self.running_cost
        self.lx_x = grad(self.lx, 0)
        self.lx_u = grad(self.lx, 1)
        self.lx_xx = jacobian(self.lx_x, 0)
        self.lx_xu = jacobian(self.lx_x, 1)
        self.lx_ux = jacobian(self.lx_u, 0)
        self.lx_uu = jacobian(self.lx_u, 1)

        # setting the terminal cost and its derivatives
        self.lf = self.terminal_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)

    
    def control(self):
        action = self.command(self.curr_state)
        self.curr_state[1] = np.mod(self.curr_state[1]+np.pi, 2*np.pi) - np.pi
        self.curr_state[2] = np.mod(self.curr_state[2]+np.pi, 2*np.pi) - np.pi
        self.curr_state = self.dynamics_analytic_np(self.curr_state, action)
        self.counter += 1
        return self.curr_state,action
        
    def calculate_error(self):
        '''calculate error between current state and goal state'''
        state_diff = (self.curr_state - self.goal_state)
        # normalize theta1, theta2 to [-pi, pi]
        state_diff[1] = np.mod(state_diff[1]+np.pi, 2*np.pi) - np.pi
        state_diff[2] = np.mod(state_diff[2]+np.pi, 2*np.pi) - np.pi
        return np.linalg.norm(np.matmul(self.error_Q, state_diff))
    

    def dynamics_analytic_np(self,state, action):
        """
        Dynamics from state_0 and taking the control action given by action for a single time step.
        :param state_0: numpy array of shape (state_size,) representing the initial state.
        :param action: numpy array of shape (1,) representing the control action.
        :return:
        * next_state: numpy array of shape (state_size,) representing the state after applying the action.
        """
        # Constants
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
       
        D = np.array([[d1,                d2*np.cos(theta1),        d3*np.cos(theta2)       ],
                        [d2*np.cos(theta1), d4,                       d5*np.cos(theta1-theta2)],
                        [d3*np.cos(theta2), d5*np.cos(theta1-theta2), d6                      ]]).astype(np.float64)
        C = np.array([[0, -d2*np.sin(theta1)*theta1_dot,        -d3*np.sin(theta2)*theta2_dot      ], 
                        [0, 0,                                    d5*np.sin(theta1-theta2)*theta2_dot], 
                        [0, -d5*np.sin(theta1-theta2)*theta1_dot, 0                                  ]]).astype(np.float64)
        G = np.array([0, -f1*np.sin(theta1), -f2*np.sin(theta2)]).astype(np.float64)
        H = np.array([1, 0, 0]).astype(np.float64)

        state_dot = state[3:] # [theta0_dot, theta1_dot, theta2_dot]
        D_inv = np.linalg.inv(D)

        accelerations = -D_inv @ C @ state_dot - D_inv @ G + D_inv @ (H * action)
        
        # Extract the accelerations (second derivatives of the states)
        [x_dot_dot,theta_1_dot_dot,theta_2_dot_dot] = accelerations
        
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
        return next_state
    
    def running_cost(self,state, action):
        '''
        :param state: np array of shape (state_dim, )
        :param action: np array of shape (action_dim, )
        :param goal_state: np array of shape (state_dim, )

        :returns cost: np array of shape (1, )
        '''
        goal_state = np.array([0., 0., 0., 0., 0., 0.])
        Q = np.diag([0.5, 10., 10., 1., 1., 1.])
        R = np.array([[0.01]])
        diff = state - goal_state
        [theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot] = diff
        theta1 = np.mod(theta1+np.pi, 2*np.pi) - np.pi
        theta2 = np.mod(theta2+np.pi, 2*np.pi) - np.pi
        diff = np.array([theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot])
        cost = diff.T @ Q @ diff + action.T @ R @ action
        return cost


    def terminal_cost(self,state):
        '''
        :param state: np array of shape (state_dim, )
        :param goal_state: np array of shape (state_dim, )

        :returns cost: np array of shape (1, )
        '''
        goal_state = np.array([0., 0., 0., 0., 0., 0.])
        Q = np.diag([1, 100, 100, 1, 1, 1])
        diff = state - goal_state
        [theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot] = diff
        theta1 = np.mod(theta1+np.pi, 2*np.pi) - np.pi
        theta2 = np.mod(theta2+np.pi, 2*np.pi) - np.pi
        diff = np.array([theta0, theta1, theta2, theta0_dot, theta1_dot, theta2_dot])
        cost = diff.T @ Q @ diff
        return cost

    def compute_cost(self,states, actions):
        '''
        :param states: np array of shape (T, state_dim)
        :param actions: np array of shape (T-1, action_dim)
        :param goal_state: np array of shape (state_dim, )

        :returns cost: integer
        '''
        cost = 0.0
        for i in range(actions.shape[0]):
            cost += self.running_cost(states[i], actions[i])
        cost += self.terminal_cost(states[-1])
        return cost
    
    def _rollout_dynamics(self, state, actions):
        '''
        :param state: np array of shape (state_dim, )
        :param actions: np array of shape (T-1, action_dim)
        :returns states: np arrary of shape (T, state_dim)
        '''
        states = [state]
        for i in range(actions.shape[0]):
            states.append(self.dynamics(states[i], actions[i]))
        states = np.stack(states)
        return states
    
    def command(self, state):
        '''
        :param state: np array of shape (state_dim, )
        :returns action: np arrary of shape (action_dim, ), which is the best action
        '''
        
        assert isinstance(state, np.ndarray)
        U = np.random.uniform(-1.0, 1.0, (self.T-1, self.action_dim)).astype(np.float64)
        # getting the intial trajectory
        X = self._rollout_dynamics(state, U)
        
        counter1 = 0
        prev_cost = 0
        curr_cost = self.compute_cost(X, U)
        mu1 = 0.01
        mu2 = 0.01
        alpha = 0.01

        # loop until convergence or max iterations
        while counter1 < self.max_iters and abs(curr_cost - prev_cost) > self.epsilon:
            # to check in Quu is PD or not
            continue_outer_loop_flag = False
            # backward pass
            # terminal cost and its derivatives
            V = self.lf(X[-1])
            Vx = self.lf_x(X[-1])
            Vxx = self.lf_xx(X[-1])

            # k and K gains
            k_gains = []
            K_gains = []
            for t in range(self.T-2, -1, -1):
                # dynamics and its gradients
                fx = self.f_x(X[t], U[t])
                fu = self.f_u(X[t], U[t])
                fuu = self.f_uu(X[t], U[t])
                fux = self.f_ux(X[t], U[t])

                # running cost and its derivatives
                lx = self.lx_x(X[t], U[t])
                lu = self.lx_u(X[t], U[t])
                lxx = self.lx_xx(X[t], U[t])
                luu = self.lx_uu(X[t], U[t])
                lux = self.lx_ux(X[t], U[t])
                
                Qx = lx + fx.T @ Vx
                Qu = lu + fu.T @ Vx
                
                Qxx = lxx + fx.T @ (Vxx + mu1 * np.eye(self.state_dim)) @ fx
                fux = np.reshape(fux, (6, 6))
                fuu = np.reshape(fuu, (6, 1))

            
                Quu = luu + fu.T @ (Vxx + mu1 * np.eye(self.state_dim)) @ fu + mu2 * np.eye(self.action_dim)
                Qux = lux + fu.T @ (Vxx + mu1 * np.eye(self.state_dim)) @ fx 
                if abs(np.linalg.det(Quu)) < 1e-6:
                    # if Quu is not PD, increase mu and break the loop
                    continue_outer_loop_flag= True
                    mu1 += 0.01
                    mu2 += 0.01
                    break
                k = -np.linalg.pinv(Quu) @ Qu
                K = -np.linalg.pinv(Quu) @ Qux
                k_gains.append(k)
                K_gains.append(K)

                
                Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

            # line search
            counter2 = 0
            eta = 1.0
            k_gains_reversed = k_gains[::-1]
            K_gains_reversed = K_gains[::-1]

            while counter2 < self.B:
                X_bar = np.zeros_like(X)
                U_bar = np.zeros_like(U)
                X_bar[0] = X[0].copy()

                for t in range(self.T-1):
                    del_x = X_bar[t] - X[t]
                    U_bar[t] = U[t] + eta * k_gains_reversed[t] + K_gains_reversed[t] @ (del_x)
                    X_bar[t + 1] = self.f(X_bar[t], U_bar[t])

                cost = self.compute_cost(X_bar, U_bar)
                if cost < curr_cost:
                    X = X_bar
                    U = U_bar
                    break
                else:
                    eta *= self.decay
                
                counter2 += 1
            
            prev_cost = curr_cost
            curr_cost = cost
            counter1 += 1
        return U[0]

