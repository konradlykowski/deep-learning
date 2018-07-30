import numpy as np
import pylab
from scipy import stats
from sklearn.datasets.samples_generator import make_regression


def gradient_descent(learning_rate, x, y, max_iter=1000):
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    for iter in range(0, max_iter):
        t0 = t0 - learning_rate * theta0_derivative(t0, t1, x, y)
        t1 = t1 - learning_rate * theta1_derivative(t0, t1, x, y)
        e = square_error(t0, t1, x, y)
        update_error_graph(e, iter)
    return t0, t1, e


def gradient_descent_numpy(alpha, x, y, num_iterations):
    theta = np.ones(2)
    x_transpose = x.transpose()
    for iter in range(0, num_iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        theta = theta - alpha * (np.dot(x_transpose, loss) / x.shape[0])
    return theta, np.sum(loss ** 2) / (2 * x.shape[0])


def theta1_derivative(t0, t1, x, y):
    z = (t0+t1*x).transpose() - y

    grad = np.gradient(z)


    print(1.0 / x.shape[0] * sum([z[i] for i in range(0,len(z))]))
    print(1.0 / x.shape[0] * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(x.shape[0])]))
    print("CZDSZCZCZC")
    return 1.0 / x.shape[0] * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(x.shape[0])])


def theta0_derivative(t0, t1, x, y):
    return 1.0 / x.shape[0] * sum([(t0 + t1 * x[i] - y[i]) for i in range(x.shape[0])])


def update_error_graph(e, iter):
    pylab.subplot(212)
    pylab.plot(iter, e, 'bo', iter, e, 'k-')
    pylab.subplot(211)


def square_error(t0, t1, x, y):
    return sum([(linear_hipothesis(i, t0, t1, x) - y[i]) ** 2 for i in range(x.shape[0])])


def linear_hipothesis(i, t0, t1, x):
    return t0 + t1 * x[i]


def update_result_graph():
    for i in range(x.shape[0]):
        y_predict = theta0 + theta1 * x
    pylab.plot(x, y_predict, 'k-')
    pylab.show()


pylab.subplot(211)

x, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35)
pylab.plot(x, y, 'o')

print('x.shape = {} y.shape = {}'.format(x.shape, y.shape))
theta0, theta1, e = gradient_descent(0.01, x, y, max_iter=1000)
print('theta0 = {} theta1 = {} error = {}'.format(theta0, theta1, e))
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:, 0], y)
print('intercept = {} slope = {}'.format(intercept, slope))

###
print('x.shape = {} y.shape = {}'.format(x.shape, y.shape))
x = np.c_[np.ones(np.shape(x)), x]  # insert column

theta2 = gradient_descent_numpy(0.01, x, y, 1000)
print('theta0 = {} theta1 = {} error = {}'.format(theta2[0][0], theta2[0][1], theta2[1]))
###

update_result_graph()
print("Done!")
