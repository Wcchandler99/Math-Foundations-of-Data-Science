import numpy as np
import matplotlib.pyplot as plt

# Generate random data
A = np.random.normal(0, 1, (20, 20))
b = np.random.normal(0, 1, (20))
a = 0.01  # Learning rate
beta = 0.9  # Momentum term

# Initialize variables
vanilla_x_current = np.random.normal(0, 1, (20))
vanilla_x_old = vanilla_x_current.copy()
vanilla_error_values = []

# Gradient descent with momentum
for t in range(50):
    ATA = np.dot(np.transpose(A), A)
    ATAx = np.dot(ATA, vanilla_x_current)
    ATb = np.dot(np.transpose(A), b)

    vanilla_x_new = vanilla_x_current - a * (2 * ATAx - 2 * ATb)
    vanilla_x_old = vanilla_x_current
    vanilla_x_current = vanilla_x_new

    # Compute the error and store it
    error = np.linalg.norm(A @ vanilla_x_current - b)
    vanilla_error_values.append(error)

momentum_x_current = np.random.normal(0, 1, (20))
momentum_x_old = momentum_x_current.copy()
momentum_error_values = []

# Gradient descent with momentum
for t in range(50):
    ATA = np.dot(np.transpose(A), A)
    ATAx = np.dot(ATA, momentum_x_current)
    ATb = np.dot(np.transpose(A), b)

    momentum_x_new = momentum_x_current - a * (2 * ATAx - 2 * ATb) + beta * (momentum_x_current - momentum_x_old)
    momentum_x_old = momentum_x_current
    momentum_x_current = momentum_x_new

    # Compute the error and store it
    error = np.linalg.norm(A @ momentum_x_current - b)
    momentum_error_values.append(error)

# Plot Error vs Steps
plt.figure(figsize=(8, 5))
plt.plot(range(len(vanilla_error_values)), vanilla_error_values, label="Vanilla Gradient Descent", color="blue")
plt.plot(range(len(momentum_error_values)), momentum_error_values, label="Momentum Gradient Descent", color="red")
plt.xlabel("Steps")
plt.ylabel("Error (||Ax - b||)")
plt.title("Error vs Steps in Gradient Descent")
plt.legend()
plt.grid()

# Save and display the plot
plt.savefig("error_vs_steps.png")
plt.show()
