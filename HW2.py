import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(6)
A = np.random.normal(0, 1, (20, 20))

v_current = np.random.normal(0, 1, (20))
v_old = v_current
#print(v)
e = [1] + [0]*19
#print(e)
cords1 = []
cords2 = []
iA = np.linalg.inv(A)
a = .008
for i in range(10000):
    ATA = np.dot(np.transpose(A), A)
    ATAv = np.dot(ATA, v_current)
    ATe = np.dot(np.transpose(A), e)
    Av = np.dot(A, v_current)
    AAT = np.dot(A, np.transpose(A))
    AAT2 = np.dot(AAT, AAT)
    AveT = np.transpose(Av-e)
    Ave = Av-e
    # numerator = np.dot(Ave, np.dot(A, np.dot(np.transpose(A),Ave)))
    # denominator = 2*np.dot(Ave, np.dot(A, np.dot(np.transpose(A), np.dot(A, np.dot(np.transpose(A), Ave)))))
    # a = numerator/denominator

    v_new = v_current - a*(2*(ATAv) - 2*(ATe))
    v_old = v_current
    v_current = v_new
    cords2.append(np.linalg.norm(iA[:,0] - v_current))
    print(np.linalg.norm(iA[:,0] - v_current))
    cords1.append(np.square(np.linalg.norm(A*v_current-e)))
    #print(np.square(np.linalg.norm(Av-e)))

# Create the plot
plt.plot(cords2, label="Norm Difference")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Norm Difference vs Iteration")

# Annotate the last value
last_value = cords2[-1]
plt.annotate(f"{last_value:.10f}", xy=(len(cords2) - 1, last_value),
             xytext=(len(cords2) - 2000, last_value + 0.5),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=10)

plt.legend()
plt.savefig("plot1.png")
plt.show()


