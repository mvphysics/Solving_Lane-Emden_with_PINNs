### Create a trainer for the fully connected network in PyTorch
from abc import ABC, abstractmethod
from typing import Tuple, List
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

from logging_config import setup_logging

# Set up logging
setup_logging()

class Trainer(ABC):
    """Abstract base class for training models.

    This class defines the structure for training models, including methods for 
    saving, loading, creating training data, calculating loss, and training.

    Methods:
        save_model: Abstract method to save the model state.
        load_model: Abstract method to load the model state.
        create_training_data: Abstract method to create training data.
        loss_calculator: Abstract method to calculate the loss.
        train: Abstract method to train the model.
        plot_loss: Abstract method to plot the training loss.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model


    def initialize_weights(self, initialization: str|None = None) -> None:
        """Initialize network weights using specified scheme.
        
        Args:
            initialization (str): Initialization scheme ('xavier', 'kaiming', or 'uniform')
            seed (int | None): Random seed for reproducibility
        """
        if initialization is None:
            return
        elif initialization not in ["xavier", "kaiming", "uniform"]:
            raise ValueError(f"Invalid initialization scheme: {initialization}. "
                             "Choose from 'xavier', 'kaiming', 'uniform' or leave it blank for default Pytorch initialization.")
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                if initialization == "xavier":
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
                elif initialization == "kaiming":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh')
                    nn.init.constant_(layer.bias, 0.0)
                elif initialization == "uniform":
                    nn.init.uniform_(layer.weight, -0.1, 0.1)
                    nn.init.uniform_(layer.bias, -0.1, 0.1)


    def save_model(self, filename: str) -> None:
        """Save the model state to a file."""
        path: str = f"./{filename}.pth"
        torch.save(self.model.state_dict(), path)


    def load_model(self, filename: str) -> None:
        """Load the model state from a file."""
        try:
            path: str = f"./{filename}.pth"
            self.model.load_state_dict(torch.load(path))
            logging.info(f"Model {filename}.pth loaded successfully.")
        except FileNotFoundError:
            logging.error(f"Model file {filename} not found. Please check the filename and try again.\n"
                                    "Do not include the .pth extension in the filename.")


    @abstractmethod
    def create_training_data(self, start: float, end: float, num_points: int, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create training data for the model."""
        pass
    

    @abstractmethod
    def create_evaluation_data(self, start: float, end: float, num_points: int, *args) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Create evaluation data for the model."""
        pass


    @abstractmethod
    def loss_calculator(self, *args) -> torch.Tensor:
        """Calculate the loss for the model."""
        pass
        

    @abstractmethod
    def train(self, *args) -> Tuple[List[float], dict]:
        """Train the model."""
        pass


    def plot_loss(self, n_epochs: int, losses: List[float], scale: str , save_plot: bool , filename: str|None) -> None:
        """Plot the training loss and save the plot localy.

        Args:
            losses (List[float]): List of loss values.
            n_epochs (int): Number of epochs.
            scale (str): Scale for the y-axis, either 'linear' or 'log'. Defaults to 'log'.
            save_plot (bool): Whether to save the plot. Defaults to False.
            filename (str): Filename to save the plot. Defaults to "loss_plot".
        """
        if filename is None:
            filename = "loss_plot"
        plt.figure(figsize=(6, 5))
        plt.plot(range(n_epochs + 1), losses, label="Loss")
        plt.scatter(losses.index(min(losses)), min(losses), color="red", label="Minimum Loss", zorder=5)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xlim(0, n_epochs)
        if scale == "log":
            plt.yscale("log")
        plt.ylim(min(losses), max(losses))
        plt.legend()
        if save_plot:
            plt.savefig(f"./{filename}.png")
            logging.info(f"Loss plot saved as {filename}.png")
        plt.show()


    @abstractmethod
    def evaluate(self, *args) -> None:
        """Evaluate the model on a dataset."""
        pass


    @abstractmethod
    def plot_results(self, *args) -> None:
        """Plot the results of the model."""
        pass
