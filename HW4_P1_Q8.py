import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(6)
A = np.random.normal(0, 1, (20, 20))


#print(v)
b = np.random.normal(0, 1, (20))
#print(e)
a_values = np.linspace(0.00001, 0.0001, 100)

avg_error_values = []
for a in a_values:
    error_values = []
    for _ in range(30):
        x_current = np.random.normal(0, 1, (20))
        x_old = x_current
        for _ in range(100):
            ATA = np.dot(np.transpose(A), A)
            ATAx = np.dot(ATA, x_current)
            ATb = np.dot(np.transpose(A), b)
            
            x_new = x_current * np.exp(-a*(2*(ATAx) - 2*(ATb)))
            x_old = x_current
            x_current = x_new
        print("Alpha: ", a, " error: ", np.linalg.norm(A@x_current-b))
        error = np.linalg.norm(A @ x_current - b)
        error_values.append(error)
    avg_error_values.append(np.mean(error_values))


# Plot Error vs Alpha
plt.plot(a_values, avg_error_values, marker='o', label="Error")
plt.xlabel("Alpha Value (a)")
plt.ylabel("Error")
plt.ylim(15, 40)
plt.title("Error vs Alpha Value")
plt.legend()
plt.grid()

# Save and display the plot
plt.savefig("exp.png")
plt.show()




