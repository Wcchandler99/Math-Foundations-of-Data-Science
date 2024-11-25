import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(6)
A = np.random.normal(0, 1, (20, 20))


#print(v)
e = np.random.normal(0, 1, (20))
#print(e)
a_values = np.linspace(0.0001, 0.02, 10)
b_values = np.linspace(0, .9, 10)
error_dict = {b: [] for b in b_values}
for b in b_values:
    for a in a_values:
        v_current = np.random.normal(0, 1, (20))
        v_old = v_current
        for _ in range(10):
            error = []
            for _ in range(100):
                ATA = np.dot(np.transpose(A), A)
                ATAv = np.dot(ATA, v_current)
                ATe = np.dot(np.transpose(A), e)
                
                v_new = v_current - a*(2*(ATAv) - 2*(ATe)) + b*(v_current - v_old)
                v_old = v_current
                v_current = v_new
            print("Beta: ", b, "Alpha: ", a, " error: ", np.square(np.linalg.norm(A@v_current-e)))
            error.append(np.linalg.norm(A @ v_current - e))
        error_dict[b].append(np.mean(error))


for b, errors in error_dict.items():
    plt.plot(a_values, errors, marker='o', label=f"Beta = {b:.5f}")


plt.xlabel("Alpha Value (a)")
plt.ylabel("Error")
plt.ylim(0, 1)
plt.title("Error vs Alpha Value for Different Beta Values")
plt.legend()
plt.grid()


plt.savefig("error_vs_alpha_beta.png")
plt.show()

