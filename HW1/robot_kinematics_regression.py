from torch import nn


class MLP(nn.Module):
    """
    Regression approximation via 3-FC NN layers.
    The network input features are one-dimensional as well as the output features.
    The network hidden sizes are 128.
    Activations are ReLU
    """
    def __init__(self):
        super().__init__()
        # --- Your code here
        input_dimension = 3
        output_dimension = 2
        hidden_layer_dimension = 128
        self.linear1 = nn.Linear(input_dimension, hidden_layer_dimension)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layer_dimension , hidden_layer_dimension)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_layer_dimension,output_dimension)
        # ---

    def forward(self, x):
        """
        :param x: Tensor of size (N, 3)
        :return: y_hat: Tensor of size (N, 2)
        """
        y_hat = None
        # --- Your code here
        y_hat = self.linear1(x)
        y_hat = self.activation1(y_hat)
        y_hat = self.linear2(y_hat)
        y_hat = self.activation2(y_hat)
        y_hat = self.linear3(y_hat)
        # ---
        return y_hat
