import numpy as np
import random
import matplotlib.pyplot as plt


A = np.random.choice([-1, 0, 1], size = (10, 10))
# Make the matrix symmetric by setting A to (A + A.T)
M = np.where(A + A.T > 0, 1, np.where(A + A.T < 0, -1, 0))

M_eigenvalues = np.linalg.eigvals(M)
M_max_eigenvalue = np.max(M_eigenvalues)
M_min_eigenvalue = np.min(M_eigenvalues)
M_spectral_radius = M_max_eigenvalue - M_min_eigenvalue
max_spectral_radius_data = []
#print(max_eigenvalue)
x = 0
for _ in range(100):
    neighbor_spectral_radius = []
    neighbors = []
    #print("M:")
    #print(M)
    for i, row in enumerate(M):
        N = M.copy()
        row_num = i
        col_num = random.randint(i, 9)
        new_entry = np.random.choice([-1, 0, 1])
        N[row_num, col_num] = new_entry
        N[col_num, row_num] = new_entry
        if np.array_equal(N, N.T):
            #print("Neighbor ", i, ":")
            #print(N)
            eigenvalues = np.linalg.eigvals(N)
            max_eigenvalue = np.max(eigenvalues)
            min_eigenvalue = np.min(eigenvalues)
            #print("Max Eigenvalue: ", max_eigenvalue)
            neighbor_spectral_radius.append(max_eigenvalue - min_eigenvalue)
            neighbors.append(N)

    #print("M max eigenvalue: ", M_max_eigenvalue)
    #print("Neighbor max eigenvalue: ", np.max(neighbor_max_eigenvalues))
    if np.max(neighbor_spectral_radius) > M_spectral_radius:
        #print("True")
        x += 1
        M = neighbors[np.argmax(neighbor_spectral_radius)]
        M_eigenvalues = np.linalg.eigvals(M)
        M_max_eigenvalue = np.max(M_eigenvalues)
        M_min_eigenvalue = np.min(M_eigenvalues)
        M_spectral_radius = M_max_eigenvalue - M_min_eigenvalue
        max_spectral_radius_data. append(np.max(neighbor_spectral_radius))


        #print("New M: ")
        #print("Number: ", np.max(neighbor_max_eigenvalues))
print(x)
print(M)
M_eigenvalues = np.linalg.eigvals(M)
M_max_eigenvalue = np.max(M_eigenvalues)
M_min_eigenvalue = np.min(M_eigenvalues)
M_spectral_radius = M_max_eigenvalue - M_min_eigenvalue
print(M_spectral_radius)
print(np.array_equal(M, M.T))
plt.plot(max_spectral_radius_data)
plt.savefig("plot.png")


    