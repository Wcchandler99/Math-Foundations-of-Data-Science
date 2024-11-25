import numpy as np
import matplotlib.pyplot as plt

beta = 1
def f2(x, beta):
    return (2 - 6 * beta * x**2)/((1 + beta * x**2)**3)


t_values = np.linspace(0, 100, 100)  
beta = .01
alpha = 1/beta

def rate_converge(t, beta, x):
    return (1-alpha*f2(0, beta))**t

y_values = rate_converge(t_values, beta)

plt.plot(t_values, y_values, label=f"β = {beta}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Plot of f(x) = 2 - 6 * beta * x**2)/((1 + beta * x**2)**3")
plt.legend()
plt.grid(True)
plt.savefig("plot.png")