import numpy as np
import pylab
import sklearn
from scipy import stats, math
from sklearn.datasets.samples_generator import make_regression


def gradient_descent(learning_rate, x, y, max_iter=1000):
    t0 = 1

    for iter in range(0, max_iter):
        print(t0)
        t0 = t0 - learning_rate * theta0_derivative(t0, x, y)

        e = square_error(t0, x, y)
        # update_error_graph(e, iter)
    return t0, e


def theta0_derivative(t0, x, y):
    sum = 0
    for i in range(x.shape[0]):
        #print(t0,"---", x[i],"aaaaaa",(t0 - x[i] ** 2), "----", math.sqrt(t0 - x[i] ** 2))
        sum = sum + ((math.sqrt(t0 - x[i] ** 2) - y[i]) / (math.sqrt(t0 - x[i] ** 2)))
    return (1.0 / 2 * x.shape[0]) * sum


def update_error_graph(e, iter):
    pylab.subplot(212)
    pylab.plot(iter, e, 'bo', iter, e, 'k-')
    pylab.subplot(211)


def square_error(t0, x, y):
    return sum([(circle_hipothesis(i, t0, x) - y[i]) ** 2 for i in range(x.shape[0])])


def circle_hipothesis(i, t0, x):
    return math.sqrt(t0 - x[i] ** 2)


def update_result_graph():
    y_predict = []
    for i in range(x.shape[0]):
        y_predict.append(math.sqrt(theta0 - x[i] ** 2))
    pylab.plot(x, y_predict, 'o')
    #pylab.show()


pylab.subplot(211)

X, _ = sklearn.datasets.make_circles(factor=0.5)
x, y = X[:, 0], X[:, 1]

pylab.plot(x, y, 'o')

print('x.shape = {} y.shape = {}'.format(x.shape, y.shape))
theta0, error = gradient_descent(0.01, x, y, max_iter=1000)
print('theta0 = {} theta1 = {} error '.format(theta0, error))

update_result_graph()
print("Done!")
