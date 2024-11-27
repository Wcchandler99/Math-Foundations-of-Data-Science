import numpy as np

# Set the random seed for reproducibility
#np.random.seed(4)

# Generate A, b, and x using NumPy
A = np.random.normal(0, 1, (20, 20))
b = np.random.normal(0, 1, (20))
x = np.random.normal(0, 1, (20))


# Evaluate the Hessian matrix at x
hessian_matrix = np.dot(np.transpose(A), A)

print("Hessian matrix:")
print(hessian_matrix)

print("EigenValues: ")
print(np.linalg.eigvals(hessian_matrix))

print("Max EigenValue: ")
print(max(np.linalg.eigvals(hessian_matrix)))

print("Max alpha: ")
print(2/max(np.linalg.eigvals(hessian_matrix)))

cond = np.linalg.cond(hessian_matrix)
print("Condition Number:", cond)


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6)
A = np.random.normal(0, 1, (20, 20))


#print(v)
b = np.random.normal(0, 1, (20))
#print(e)
a_values = [.018] #np.linspace(0.01, .04, 8)

avg_error_values = []
for a in a_values:
    error_values = []
    for _ in range(1):
        x = np.random.normal(0, 1, (20))
        for _ in range(100):
            ATA = np.dot(np.transpose(A), A)
            ATAx = np.dot(ATA, x)
            ATb = np.dot(np.transpose(A), b)
            
            x = x - a*(2*(ATAx) - 2*(ATb))
            print("Alpha: ", a, " error: ", np.square(np.linalg.norm(A@x-b)))
        error = np.linalg.norm(A @ x - b)
        error_values.append(error)
    avg_error_values.append(np.mean(error_values))


# Plot Error vs Alpha
plt.plot(a_values, avg_error_values, marker='o', label="Error")
plt.xlabel("Alpha Value (a)")
plt.ylabel("Error")
plt.ylim(0, .6)
plt.title("Error vs Alpha Value")
plt.legend()
plt.grid()

# Save and display the plot
plt.savefig("error_vs_max_alpha.png")
plt.show()