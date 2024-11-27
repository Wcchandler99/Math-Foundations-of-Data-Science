import autograd.numpy as np
from autograd import hessian

# Set the random seed for reproducibility
np.random.seed(6)

# Generate A, b, and x using NumPy
A = np.random.normal(0, 1, (20, 20))
b = np.random.normal(0, 1, (20))
x = np.random.normal(0, 1, (20))

# Define your function
def my_function(x):
    return np.linalg.norm(A @ x - b)

# Calculate the Hessian as a callable function
hessian_function = hessian(my_function)

# Evaluate the Hessian matrix at x
hessian_matrix = hessian_function(x)

print("Hessian matrix:")
print(hessian_matrix)

print("EigenValues: ")
print(np.linalg.eigvals(hessian_matrix))

print("Max EigenValue: ")
print(max(np.linalg.eigvals(hessian_matrix)))