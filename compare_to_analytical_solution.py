# For n=1 case, compare PINN solution to analytical solution
import torch
import numpy as np
import matplotlib.pyplot as plt

from nn_architecture import FCN
from single_case_trainer import SingleCaseTraniner

# Analytical solution for Lane-Emden n=1
def analytical_solution_n1(xi):
    return np.sin(xi) / xi

# Load trained PINN model
n = 1
pinn = FCN(n_input=1, n_output=1, n_hidden=64, n_layers=4)
trainer = SingleCaseTraniner(pinn, n)
trainer.load_model("PINNn1")

# Evaluation points
xi = torch.linspace(0, 10, 100).view(-1, 1)
theta_pinn = trainer.evaluate(xi)
theta_analytical = analytical_solution_n1(xi.numpy().flatten())

# Plot comparison
plt.figure(figsize=(8, 6))
plt.plot(xi.numpy(), theta_pinn.numpy(), label="PINN Solution", color="tab:blue")
plt.plot(xi.numpy(), theta_analytical, label="Analytical Solution", color="tab:red", linestyle="--")
plt.xlabel("xi")
plt.ylabel("theta")
plt.title("PINN vs Analytical Solution for Lane-Emden (n=1)")
plt.legend()
plt.grid()
plt.savefig("PINN_vs_Analytical_n1.png")
plt.show()