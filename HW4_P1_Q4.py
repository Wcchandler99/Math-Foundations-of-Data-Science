import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(6)
A = np.random.normal(0, 1, (20, 20))


#print(v)
b = np.random.normal(0, 1, (20))
#print(e)
a = .01
beta_values = np.linspace(0, 1, 100)

avg_error_values = []
for beta in beta_values:
    error_values = []
    for _ in range(10):
        x_current = np.random.normal(0, 1, (20))
        x_old = x_current
        for _ in range(1000):
            ATA = np.dot(np.transpose(A), A)
            ATAx = np.dot(ATA, x_current)
            ATb = np.dot(np.transpose(A), b)
            
            x_new = x_current - a*(2*(ATAx) - 2*(ATb)) + beta*(x_current - x_old)
            x_old = x_current
            x_current = x_new
        print("Beta: ", beta, " error: ", np.square(np.linalg.norm(A@x_current-b)))
        error = np.linalg.norm(A @ x_current - b)
        error_values.append(error)
    avg_error_values.append(np.mean(error_values))


# Plot Error vs Alpha
plt.plot(beta_values, avg_error_values, marker='o', label="Error")
plt.xlabel("Beta Value (b)")
plt.ylabel("Error")
plt.ylim(0, .8)
plt.title("Error vs Beta Value")
plt.legend()
plt.grid()

# Save and display the plot
plt.savefig("error_vs_beta.png")
plt.show()