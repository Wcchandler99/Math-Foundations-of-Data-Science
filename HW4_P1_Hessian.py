import sympy as sp

# Define variables
n = 3  # Number of variables (example size, adjust as needed)
x = sp.Matrix(sp.symbols(f'x1:{n+1}'))  # Vector of variables
A = sp.MatrixSymbol('A', n, n)  # Matrix A
b = sp.MatrixSymbol('b', n, 1)  # Vector b

# Define F(x) = ||Ax - b||^2
Ax_b = sp.Matrix(A) * x - sp.Matrix(b)
F = (Ax_b.T * Ax_b)[0]

# Compute the Hessian matrix
Hessian = sp.hessian(F, x)

# Print the result
sp.pprint(Hessian)
