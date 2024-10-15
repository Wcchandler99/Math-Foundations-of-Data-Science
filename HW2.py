import numpy as np
import pandas
import matplotlib.pyplot as plt

np.random.seed(3)
A = np.random.normal(0, 1, (20, 20))

v = np.random.normal(0, 1, (20))
#print(v)
e = [1] + [0]*19
#print(e)
cords1 = []
cords2 = []
iA = np.linalg.inv(A)
a = .01
for i in range(100):
    ATA = np.dot(np.transpose(A), A)
    ATAv = np.dot(ATA, v)
    ATe = np.dot(np.transpose(A), e)
    Av = np.dot(A, v)
    AAT = np.dot(A, np.transpose(A))
    AAT2 = np.dot(AAT, AAT)
    AveT = np.transpose(Av-e)
    Ave = Av-e
    # numerator = np.dot(Ave, np.dot(A, np.dot(np.transpose(A),Ave)))
    # denominator = 2*np.dot(Ave, np.dot(A, np.dot(np.transpose(A), np.dot(A, np.dot(np.transpose(A), Ave)))))
    # a = numerator/denominator

    v = v - a*(2*(ATAv) - 2*(ATe))

    cords2.append(np.linalg.norm(iA[0] - v))
    print(np.linalg.norm(iA[0] - v))
    cords1.append(np.square(np.linalg.norm(A*v-e)))
    #print(np.square(np.linalg.norm(Av-e)))



plt.plot(cords2)
plt.savefig("plot1.png")
