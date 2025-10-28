import numpy as np

def train(x, y, iterations, learning_rate):
    w = 0.0
    b = 0.0
    lam = 0.1
    m = len(x)
    for _ in range(iterations):
        y_predict = w * x + b
        dw = (1/m) * np.dot((y_predict - y), x) + (lam/m) * w
        db = (1/m) * np.sum(y_predict - y)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w,b

def main():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 7, 9, 11])
    learning_rate = 0.01
    iterations = 1000
    w,b = train(x, y, iterations, learning_rate)
    test_x = np.array([6, 7, 8])
    prediction = w*test_x + b
    print(prediction)

if __name__ == "__main__":
    main()