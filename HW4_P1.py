import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(6)
A = np.random.normal(0, 1, (20, 20))


#print(v)
e = np.random.normal(0, 1, (20))
#print(e)
a_values = np.linspace(0.0001, 0.015, 100)

avg_error_values = []
for a in a_values:
    error_values = []
    for _ in range(10):
        v_current = np.random.normal(0, 1, (20))
        v_old = v_current
        for _ in range(1000):
            ATA = np.dot(np.transpose(A), A)
            ATAv = np.dot(ATA, v_current)
            ATe = np.dot(np.transpose(A), e)
            
            v_new = v_current - a*(2*(ATAv) - 2*(ATe))
            v_old = v_current
            v_current = v_new
        print("Alpha: ", a, " error: ", np.square(np.linalg.norm(A@v_current-e)))
        error = np.linalg.norm(A @ v_current - e)
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
plt.savefig("error_vs_alpha.png")
plt.show()

