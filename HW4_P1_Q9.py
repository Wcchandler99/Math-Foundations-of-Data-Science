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
for t in range(1000):
    ATA = np.dot(np.transpose(A), A)
    ATAx = np.dot(ATA, vanilla_x_current)
    ATb = np.dot(np.transpose(A), b)

    vanilla_x_new = vanilla_x_current - a * (2 * ATAx - 2 * ATb)
    vanilla_x_old = vanilla_x_current
    vanilla_x_current = vanilla_x_new

    # Compute the error and store it
    error = np.linalg.norm(A @ vanilla_x_current - b)
    vanilla_error_values.append(error)

e_x_current = np.random.normal(0, 1, (20))
e_x_old = e_x_current.copy()
e_error_values = []

a = .000001
# Gradient descent with momentum
for t in range(1000):
    ATA = np.dot(np.transpose(A), A)
    ATAx = np.dot(ATA, e_x_current)
    ATb = np.dot(np.transpose(A), b)

    e_x_new = e_x_current * np.exp(-a*(2*(ATAx) - 2*(ATb)))
    e_x_old = e_x_current
    e_x_current = e_x_new

    # Compute the error and store it
    error = np.linalg.norm(A @ e_x_current - b)
    e_error_values.append(error)

# Plot Error vs Steps
plt.figure(figsize=(8, 5))
plt.plot(range(len(vanilla_error_values)), vanilla_error_values, label="Vanilla Gradient Descent", color="blue")
plt.plot(range(len(e_error_values)), e_error_values, label="Exp Gradient Descent", color="red")
plt.xlabel("Steps")
plt.ylabel("Error (||Ax - b||)")
plt.title("Error vs Steps in Gradient Descent")
plt.legend()
plt.grid()

# Save and display the plot
plt.savefig("exp_error_vs_steps.png")
plt.show()
