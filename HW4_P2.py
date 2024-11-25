import numpy as np
import matplotlib.pyplot as plt

N = 500
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

X_train = X[:300]
y_train = y[:300]
N_train = 300
X_test = X[300:]
y_test = y[300:]
N_test = 200

lambda_values = []
avg_test_errors = []
#np.random.seed(6)
lambda_values = np.linspace(0, 1, 10)
for lambda_test in lambda_values:
    test_errors = []
    for l in range(10):
        w = np.random.normal(0, 1, (101))
        #print(e)
        cords1 = []
        cords2 = []
        a = .003
        for i in range(100):
            XTX = np.dot(np.transpose(X_train), X_train)
            XTXw = np.dot(XTX, w)
            XTy = np.dot(np.transpose(X_train), y_train)
            Xw = np.dot(X_train, w)
            XXT = np.dot(X_train, np.transpose(X_train))
            XXT2 = np.dot(XXT, XXT)
            XwyT = np.transpose(Xw-y_train)
            Xwy = Xw-y_train
            # numerator = np.dot(Ave, np.dot(A, np.dot(np.transpose(A),Ave)))
            # denominator = 2*np.dot(Ave, np.dot(A, np.dot(np.transpose(A), np.dot(A, np.dot(np.transpose(A), Ave)))))
            # a = numerator/denominator

            w = w - a*((1/N_train)*(2*(XTXw) - 2*(XTy)) + 2*w*(lambda_test/101))
            #cords1.append((1/N_train)*np.linalg.norm(np.dot(X_train, w) - y_train))
            #print((1/N_train)*np.linalg.norm(np.dot(X_train, w) - y_train))

        test_errors.append((1/N_test)*np.linalg.norm(np.dot(X_test, w) - y_test))
        print("Lambda: ", lambda_test, " test error: ", (1/N_test)*np.linalg.norm(np.dot(X_test, w) - y_test))
    avg_test_errors.append(np.mean(test_errors))
# Plot test errors across lambda_test values
plt.plot(lambda_values, avg_test_errors, marker='o', label="Test Error")
plt.xlabel("Lambda Value (lambda_test)")
plt.ylabel("Test Error")
plt.title("Test Error vs Lambda Value")
plt.legend()
plt.grid()
plt.savefig("test_error_vs_lambda.png")
plt.show()


