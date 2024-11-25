import numpy as np
import matplotlib.pyplot as plt

N = 100
X = []
y = []
for j in range(N):
    #xy = []
    x = []
    x.append(1)
    for _ in range(50):
        x.append(np.random.normal(0, 1))

    for i in range(10):
        x.append(x[1]+(.5*x[i+1] + np.random.normal(0, .1)))

    for i in range(10):
        x.append(x[i+1] - x[i+11] + x[i+21] + np.random.normal(0, .1))

    for i in range(10):
        x.append(x[6*(i+1)] + 3*x[i+1] + np.random.normal(0, .1))

    for i in range(10):
        x.append(5-x[i+71])
        
    for i in range(10):
        x.append(.5*x[50+(i+1)*4] + .5*x[50+(i+1)*3] + np.random.normal(0, .1)) 
    
    mid = []
    for i in range(50):
        mid.append(pow((-.88),(i+1)) * x[2*(i+1)])
        
    y.append(sum(mid) + np.random.normal(0, .01))

    #xy.append(x)
    #xy.append(y)
    X.append(x)
    
X_train = X[:50]
y_train = y[:50]
N_train = 50
X_test = X[50:]
y_test = y[50:]
N_test = 50

    
avg_test_errors = []

k_values = np.linspace(50, 200, 151)
for k in k_values:
    test_errors = []
    for _ in range(30):
        A = np.random.normal(0, 1, (int(k), 101))
        w = np.random.normal(0, 1, (int(k)))
        X_train_proj = np.dot(A, np.transpose(X_train)).T  
        X_test_proj = np.dot(A, np.transpose(X_test)).T   
        #print(e)
        cords1 = []
        cords2 = []
        a = .00001
        for i in range(1000):
            XTX = np.dot(np.transpose(X_train_proj), X_train_proj)
            XTXw = np.dot(XTX, w)
            XTy = np.dot(np.transpose(X_train_proj), y_train)
            Xw = np.dot(X_train_proj, w)
            XXT = np.dot(X_train_proj, np.transpose(X_train_proj))
            XXT2 = np.dot(XXT, XXT)
            XwyT = np.transpose(Xw-y_train)
            Xwy = Xw-y_train


            w = w - a*(1/N_train)*(2*(XTXw) - 2*(XTy))

        test_errors.append((1/N_test)*np.linalg.norm(np.dot(X_test_proj, w) - y_test))
        print("k: ", k, " test error: ", (1/N_test)*np.linalg.norm(np.dot(X_test_proj, w) - y_test))
    avg_test_errors.append(np.mean(test_errors))


# Plot test errors across lambda_test values
plt.plot(k_values, avg_test_errors, marker='o', label="Test Error")
plt.xlabel("k Value")
plt.ylabel("Test Error")
plt.title("Test Error vs k Value")
plt.legend()
plt.grid()
plt.savefig("test_error_vs_k.png")
plt.show()



