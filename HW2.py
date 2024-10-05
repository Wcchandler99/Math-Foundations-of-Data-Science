import numpy as np
import pandas
import matplotlib.pyplot as plt

A = np.random.normal(0, 1, (20, 20))

v = np.random.normal(0, 1, (20))
#print(v)
e = [1] + [0]*19
#print(e)
a = .001
cords1 = []
cords2 = []
iA = np.linalg.inv(A)
for i in range(10000):
    ATA = np.dot(np.transpose(A), A)
    ATAv = np.dot(ATA, v)
    ATe = np.dot(np.transpose(A), e)
    v = v - a*(2*(ATAv) - 2*(ATe))

    cords2.append(np.linalg.norm(iA[0] - v))
    print(np.linalg.norm(iA[0] - v))
    cords1.append(np.square(np.linalg.norm(A*v-e)))
    Av = np.dot(A, v)
    #print(np.square(np.linalg.norm(Av-e)))



plt.plot(cords2)
plt.show()
