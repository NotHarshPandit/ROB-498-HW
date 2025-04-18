import control
import numpy as np
import scipy.linalg
import cvxpy as cp


class LinearMPC:

    def __init__(self, A, B, Q, R, horizon):
        self.dx = A.shape[0]
        self.du = B.shape[1]
        assert A.shape == (self.dx, self.dx)
        assert B.shape == (self.dx, self.du)
        self.H = horizon
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def compute_SM(self):
        """
        Computes the S and M matrices as defined in the ipython notebook

        All the variables you need should be class member variables already

        Returns:
            S: np.array of shape (horizon * dx, horizon * du) S matrix
            M: np.array of shape (horizon * dx, dx) M matrix

        """
        S, M = None, None

        # --- Your code here
        # Initialize S and M as zero matrices
        S = np.zeros((self.H * self.dx, self.H * self.du))
        M = np.zeros((self.H * self.dx, self.dx))

        # Fill in the S and M matrices
        for i in range(self.H):
            for j in range (i+1):
                S[i*self.dx:(i+1)*self.dx, j*self.du:(j+1)*self.du] =np.matmul(np.linalg.matrix_power(self.A, i-j),self.B)
        # ---
        for i in range (self.H):
            M[i*self.dx:(i+1)*self.dx] = np.linalg.matrix_power(self.A, i+1)
        return S, M

    def compute_Qbar_and_Rbar(self):
        Q_repeat = [self.Q] * self.H
        R_repeat = [self.R] * self.H
        return scipy.linalg.block_diag(*Q_repeat), scipy.linalg.block_diag(*R_repeat)

    def compute_finite_horizon_lqr_gain(self):
        """
            Compute the controller gain G0 for the finite-horizon LQR

        Returns:
            G0: np.array of shape (du, dx)

        """
        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()

        G0 = None

        # --- Your code here
        G0 = -np.linalg.inv(Rbar + S.T @ Qbar @ S) @ S.T @ Qbar @ M
        G0 = G0[0:self.du, :]  # Extract the first du rows to match the shape (du, dx)
        # ---

        return G0

    def compute_lqr_gain(self):
        """
            Compute controller gain G for infinite-horizon LQR
        Returns:
            Ginf: np.array of shape (du, dx)

        """
        Ginf = None
        theta_T_theta, _, _ = control.dare(self.A, self.B, self.Q, self.R)

        # --- Your code here
        Ginf = -np.linalg.inv(self.R + self.B.T @ theta_T_theta @ self.B) @ self.B.T @ theta_T_theta @ self.A
        # ---
        return Ginf

    def lqr_box_constraints_qp_shooting(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing with shooting

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls

        """

        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()
        U = None
        # --- Your code here
        U = cp.Variable((self.H,self.du))
        x0 = x0.reshape(-1, 1)
        # Define the objective function
        objective = cp.Minimize(cp.quad_form(np.matmul(S,U) + np.matmul(M,x0), Qbar)+cp.quad_form(U, Rbar))

        # Define control constraints
        constraints = [
        U >= u_min,  
        U <= u_max 
        ]
        qp_problem = cp.Problem(objective, constraints)   
        qp_problem.solve()
        U = U.value
        # ---

        return U

    def lqr_box_constraints_qp_collocation(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing
            with collocation

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls
            X: np.array of shape (horizon, dx) containing sequence of optimal states

        """

        X, U = None, None

        # --- Your code here

        # Define optimization variables
        U = cp.Variable((self.H, self.du))  # Control sequence (H, du)
        X = cp.Variable((self.H + 1, self.dx))  # State sequence (H+1, dx)

        # Define cost function
        Q, R = self.Q, self.R
        objective = cp.Minimize(sum(cp.quad_form(X[i+1], Q) + cp.quad_form(U[i], R) for i in range(self.H)))

        # Define constraints
        constraints = [X[0] == x0]   
        constraints += [X[t+1] == self.A @ X[t] + self.B @ U[t] for t in range(self.H)]
        constraints+=[U <= np.tile(u_max, (self.H, 1)), U>= np.tile(u_min,(self.H,1))]
        # control constraints

        # Solve the QP
        qp_problem = cp.Problem(objective, constraints)
        qp_problem.solve()
        # 
        return U.value, X.value[1:,:] # Return only the first H states
