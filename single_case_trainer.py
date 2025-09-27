### Create a trainer for the fully connected network in PyTorch for a specific polytropic index case
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List

from nn_architecture import FCN
from trainer import Trainer
from logging_config import setup_logging

# Set up logging
setup_logging()
logging.info("-------> Running SingleCaseTrainer for Lane-Emden PINN project. <-------")

class SingleCaseTraniner(Trainer):
    """Trainer class for the Fully Connected Network (FCN) for a specific polytropic index case."""


    def __init__(self, model: FCN, n: float | int, initialization: str | None = None) -> None:
        """
        Initialize the SingleCaseTrainer.
        Args:
            model (FCN): The neural network model to train.
            n (float | int): The polytropic index.
            initialization (str | None): Weight initialization scheme ('xavier', 'kaiming', 'uniform', or None).
        """
        super().__init__(model)
        try:
            self.initialize_weights(initialization=initialization)
        except ValueError as e:
            print(f"Invalid initialization scheme: {initialization}. Error: {e}. Using default initialization.")
        self.n = n


    def create_training_data(self, start: float = 0,  end: float = 10, num_points: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create training data for the model.
        Args:
            start (float): The start value for the xi_boundary domain.
            end (float): The end value for the xi_physics domain.
            num_points (int): Number of points in the domain.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Boundary and physics domain tensors.
        """
        xi_boundary = torch.tensor(0.0).view(-1, 1).requires_grad_(True)
        xi_physics = torch.linspace(start, end, num_points).view(-1, 1).requires_grad_(True)
        return xi_boundary, xi_physics


    def create_evaluation_data(self, start:float = 0, end: float = 10, num_points: int = 100) -> torch.Tensor:
        """
        Create evaluation data for the model.
        Args:
            start (float): The start value for the xi_physics domain.
            end (float): The end value for the xi_physics domain.
            num_points (int): Number of points in the domain.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Boundary and physics domain tensors.
        """
        xi_eval = torch.linspace(start, end, num_points).view(-1, 1)
        return xi_eval


    def loss_calculator(self, xi_boundary: torch.Tensor, xi_physics: torch.Tensor, 
                        lambda1: float = 1, lambda2: float = 1, lambda3: float = 0.01) -> torch.Tensor:
        """
        Compute the PINN loss function.
        Args:
            xi_boundary (torch.Tensor): Boundary input tensor.
            xi_physics (torch.Tensor): Physics domain input tensor.
            lambda1 (float): Weight for boundary loss.
            lambda2 (float): Weight for derivative boundary loss.
            lambda3 (float): Weight for physics loss.
        Returns:
            torch.Tensor: The total loss.
        """
        lambda1, lambda2, lambda3 = 1, 1, 0.01

        # Boundary loss
        theta = self.model(xi_boundary)
        loss1 = (torch.squeeze(theta) - 1)**2
        dthetadxi = torch.autograd.grad(theta, xi_boundary, torch.ones_like(theta), create_graph=True)[0]
        loss2 = (torch.squeeze(dthetadxi) - 0)**2
        
        # compute physics loss
        theta = self.model(xi_physics)
        dthetadxi = torch.autograd.grad(theta, xi_physics, torch.ones_like(theta), create_graph=True)[0]
        d2thetadxi2 = torch.autograd.grad(dthetadxi, xi_physics, torch.ones_like(dthetadxi), create_graph=True)[0]
        # Only clamp if n is not integer
        if float(self.n).is_integer():
            theta_pow = theta ** self.n
        else:
            theta_pow = torch.pow(torch.complex(theta, torch.zeros_like(theta)), self.n).real
        loss3 = torch.mean((2*dthetadxi + xi_physics*d2thetadxi2 + xi_physics*theta_pow)**2)
        return lambda1*loss1 + lambda2*loss2 + lambda3*loss3 


    def train(self, xi_boundary: torch.Tensor, xi_physics: torch.Tensor, n_epochs: int, lr: float = 1e-3,
               plot_scale: str = "log", save_plot: bool = False, modelfile: str|None = None) -> Tuple[List[float], dict]:
        """Train the model.
        Args:
            xi_boundary (torch.Tensor): Boundary input tensor.
            xi_physics (torch.Tensor): Physics domain input tensor.
            n_epochs (int): Number of epochs.
            lr (float): Learning rate.
            plot_scale (str): Scale for the loss plot, either 'linear' or 'log'.
            save_plot (bool): Whether to save the loss plot.
            modelfile (str | None): Filename to save the model.
        Returns:
            list[float]: List of loss values during training.
        """
        if modelfile is None:
            modelfile = f"PINNn{str(self.n)}"

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        losses: list[float] = []
        min_loss = float('inf')
        logging.info(f"Best model will be saved as {modelfile}.pth")
        
        logging.info(f"Starting training for {n_epochs} epochs with learning rate {lr} and model file {modelfile}")
        for epochi in range(n_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            loss = self.loss_calculator(xi_boundary, xi_physics)
            loss.backward()
            losses.append(loss.item())
            if loss.item() < min_loss:
                min_loss = loss.item()
                self.save_model(modelfile)
                best_epoch = epochi
            optimizer.step()
            if epochi % 500 == 0:
                logging.info(f"Epoch: {epochi} =====> Current Loss: {loss.item()} | Minimum Loss: {min_loss}")
                self.scheduler.step() # Step the scheduler every 500 epochs

        self.model.eval()
        loss_plot_name = modelfile + "_loss_plot"
        self.plot_loss(n_epochs, losses ,scale=plot_scale, save_plot=save_plot, filename=loss_plot_name)
        self.load_model(modelfile)  # Load the best model after training
        logging.info(f"Training completed. Minimum loss: {min_loss} at epoch {best_epoch}. Model saved as {modelfile}.pth")

        return losses

    def evaluate(self, xi: torch.Tensor, modelfile: str|None = None) -> torch.Tensor:
        """
        Evaluate the model on a given input tensor.
        Args:
            xi (torch.Tensor): Input tensor for evaluation.
            modelfile (str|None): Filename to load the model from.
        Returns:
            torch.Tensor: Output tensor from the model.
        """
        try:
            if modelfile is not None:
                self.load_model(modelfile)
            else:
                modelfile = f"PINNn{str(self.n)}"
            if not os.path.isfile(f"{modelfile}.pth"):
                raise FileNotFoundError
            self.model.eval()
            logging.info(f"Evaluating model {modelfile} on input tensor.")
            theta = self.model(xi.view(-1, 1)).detach()
            logging.info(f"Model evaluation completed.")
        except FileNotFoundError:
            logging.error(f"Model file '{modelfile}.pth' not found. Please check the filename and try again or train the model first.")
            theta = None
        return theta

    def plot_results(self, xi: torch.Tensor, theta: torch.Tensor, 
                     save_plot: bool = False, filename: str|None = None) -> None:
        """
        Plot the results of the model.
        Args:
            xi (torch.Tensor): Input tensor for plotting.
            theta (torch.Tensor): Output tensor from the model.
            save_plot (bool): Whether to save the plot.
            filename (str|None): Filename to save the plot.
        """
        try:
            if filename is None:
                filename = f"./results_plot_n{self.n}.png"
            plt.figure(figsize=(10, 8))
            plt.plot(xi, theta, label="PINN solution", color="tab:blue")
            plt.grid()
            plt.title(f'Final Result for n={self.n}')
            plt.xlabel('xi')
            plt.ylabel('theta')
            plt.xlim(xi[0], xi[-1])
            # plt.ylim(min(theta), max(theta))
            plt.ylim(-1, 1)
            if save_plot:
                plt.savefig(filename)
                logging.info(f"Results plot saved as {filename}")
            plt.show()
        except ValueError as e:
            logging.error(f"Error in plotting results: {e}. Ensure that xi and theta are compatible for plotting.")


if __name__ == "__main__":
    """
    Example usage of the SingleCaseTrainer class with a Fully Connected Network (FCN) model.
    """
    n_epochs: int = 15000
    results: list = []
    epochs: list[int] = [epoch for epoch in range(n_epochs + 1)]
    n: float = 1
    torch.manual_seed(42)  # Set random seed for reproducibility
    pinn = FCN(n_input=1, n_output=1, n_hidden=64, n_layers=4)
    trainer = SingleCaseTraniner(pinn, n, initialization=None)
    xi_boundary, xi_physics = trainer.create_training_data(end=10, num_points=200)  # Create training data for the model
    trainer.train(xi_boundary=xi_boundary, xi_physics=xi_physics, n_epochs=n_epochs, lr=6e-3, save_plot=True)
    xi = trainer.create_evaluation_data(end = 10, num_points= 100)  # Create evaluation data for the model
    trainer.load_model(f"PINNn{str(n)}")
    trainer.plot_results(xi, trainer.evaluate(xi),save_plot=True)  # Plot results
