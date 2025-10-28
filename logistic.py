import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def train(x, y, learning_rate, iterations):
    m, n = np.shape(x)
    w = np.zeros(n)
    b = 0.0
    lam = 0.1
    for _ in range(iterations):
        y_precit = sigmoid(np.dot(x, w) + b)
        dx = (1/m) * np.dot(x.T, (y_precit - y)) + (lam/m) * w
        dy = (1/m) * np.sum(y_precit - y)
        w -= learning_rate * dx
        b -= learning_rate * dy
    return w, b

def main():
    # Features: [hours studied, hours slept]
    x_train = np.array([
        [1, 5],   
        [2, 4],
        [3, 6],
        [4, 4],
        [5, 7],
        [1, 2],
        [2, 1],
        [6, 8],
        [7, 3],
        [8, 5]
    ])

    y_train = y_train = np.array([0,0,1,1,1,0,0,1,1,1])

    learning_rate = 0.1
    iterations = 1000

    w,b = train(x_train, y_train, learning_rate, iterations)

    x_test = np.array([
        [3, 5],   # medium study/sleep
        [1, 1],   # low study/sleep
        [6, 6],   # high study/sleep
        [10, 3]
    ])

    predict = sigmoid(np.dot(x_test,w) + b)
    print(predict)
    print(predict >= 0.5)

if __name__ == "__main__":
    main()