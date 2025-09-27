import torch
from torch import nn
from typing import Callable


class FCN(nn.Module):
    """Defines a standard fully-connected network in PyTorch.

    This class implements a fully-connected neural network with customizable 
    input size, output size, hidden layer size, number of layers, and activation 
    function. The network uses dropout layers for regularization.

    Attributes:
        fcs (nn.Sequential): The first layer of the network, consisting of 
            an input layer followed by the specified activation function.
        fch (nn.Sequential): The hidden layers of the network, consisting of 
            multiple layers with the specified activation function and dropout.
        fce (nn.Linear): The final layer of the network, mapping the hidden 
            layer outputs to the desired output size.

    Args:
        n_input (int): The size of the input features.
        n_output (int): The size of the output features.
        n_hidden (int): The size of the hidden layers.
        n_layers (int): The number of hidden layers in the network.
        activation (Callable): The activation function to use in the network. 
            Defaults to `nn.Tanh`.
    """
    
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int, activation: Callable = nn.Tanh) -> None:
        super().__init__()
        activation = activation
        self.fcs: nn.Sequential = nn.Sequential(*[
            nn.Linear(n_input, n_hidden),
            activation()
        ])
        self.fch: nn.Sequential = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                activation()
            ], nn.Dropout(0)) for _ in range(n_layers - 1)
        ])
        self.fce: nn.Linear = nn.Linear(n_hidden, n_output)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
    
    