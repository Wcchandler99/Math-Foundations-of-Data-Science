import numpy as np
import matplotlib.pyplot as plt

beta = 1
def f(x, beta):
    return 1 - (11 / (1 + beta * x**2))

x_values = np.linspace(-10, 10, 400)  
beta = .01

y_values = f(x_values, beta)

plt.plot(x_values, y_values, label=f"β = {beta}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Plot of f(x) = 1 - 11 / (1 + β x^2)")
plt.legend()
plt.grid(True)
plt.savefig("plot.png")