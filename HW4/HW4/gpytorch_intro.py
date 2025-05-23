import tqdm
import torch
import gpytorch
import matplotlib.pyplot as plt

from gpytorch.kernels import RBFKernel, CosineKernel, LinearKernel, PolynomialKernel, MaternKernel, ScaleKernel


class RBF_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        # --- Your code here

        # ---

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PolynomialGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, degree=4):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        # --- Your code here

        # ---

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LinearCosineGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None
        # --- Your code here

        # ---

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_hyperparams(model, likelihood, train_x, train_y, lr):
    """
        Function which optimizes the GP Kernel & likelihood hyperparameters
    Args:
        model: gpytorch.model.ExactGP model
        likelihood: gpytorch likelihood
        train_x: (N, dx) torch.tensor of training inputs
        train_y: (N, dy) torch.tensor of training targets
        lr: Learning rate

    """

    # --- Your code here

    # ---


def plot_gp_predictions(model, likelihood, train_x, train_y, test_x, title):
    """
        Generates GP plots for GP defined by model & likelihood
    """
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        lower, upper = observed_pred.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-10, 10])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(title)
        plt.show()
