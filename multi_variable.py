import numpy as np

def train(x, y, learning_rate, iterations):
    m, n = np.shape(x)
    w = np.zeros(n)
    b = 0.0
    for _ in range(iterations):
        y_precit = np.dot(x, w) + b
        dx = (1/m) * np.dot(x.T, (y_precit - y))
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
        [4, 5],
        [5, 7]
    ])
    y_train = np.array([50, 55, 65, 70, 80])

    learning_rate = 0.01
    iterations = 1000

    w,b = train(x_train, y_train, learning_rate, iterations)

    x_test = np.array([
        [6, 8],
        [3, 3]
    ])
    predict = np.dot(x_test,w) + b
    print(predict)

if __name__ == "__main__":
    main()