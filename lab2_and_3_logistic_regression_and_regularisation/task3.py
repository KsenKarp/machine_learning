import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(50)
path = 'ex2data2.txt'
df = pd.read_csv(path, header=None,
                 names=['1st_test', '2nd_test', 'decision_on_correspondence'])

X0 = np.ones((df['1st_test'].size, 3))
X0[:, 1] = df['1st_test'].values
X0[:, 2] = df['2nd_test'].values

# вектор выходных данных
y = np.array(df['decision_on_correspondence']).reshape(-1, 1)

max_degree = 30
X_strike = np.column_stack([X0[:, 1] ** i * X0[:, 2] ** (degree - i) for degree in range(max_degree + 1)
                            for i in range(degree + 1)])

# Разбиение данных на тренировочный, кроссвалидационный и тестовый наборы данных
idx = np.arange(len(X_strike))
np.random.shuffle(idx)

split_idx1 = int(0.6 * len(X_strike))
split_idx2 = int(0.8 * len(X_strike))

X_train, X_cv, X_test = X_strike[idx[:split_idx1]], X_strike[idx[split_idx1:split_idx2]], X_strike[idx[split_idx2:]]
y_train, y_cv, y_test = y[idx[:split_idx1]], y[idx[split_idx1:split_idx2]], y[idx[split_idx2:]]


def g(z):
    return np.exp(z) / (1 + np.exp(z))


def h_theta(theta, X):
    return g(np.dot(X, theta))


eps = 1e-9


def L(theta, lambda_, X, y):
    loss = 0
    for i in range(X.shape[0]):
        loss -= ((y[i, 0] * np.log(h_theta(theta, X[i]) + eps) + (1 - y[i, 0]) * np.log(1 - h_theta(theta, X[i]) + eps))
                 / X.shape[0])
    for i in range(X.shape[1]):
        loss += lambda_ * theta[i] ** 2 / (2 * X.shape[0])
    return loss


alpha = 0.5
iterations_num = 1000


def grad(theta, lambda_, X, y):
    grad = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        grad += (h_theta(theta, X[i]) - y[i, 0]) * X[i]
    if lambda_ != 0:
        grad += theta
    return grad


# Обучение и выбор значения параметра регуляризации, при котором обученная модель достигает минимального значения функции потерь (нерегуляризованной!) на кроссвалидационной выборке

def train_model(X_train, y_train, X_cv, y_cv, reg_param_range, reg_param_step):

    min_loss = float('inf')
    min_reg_param = 0.0

    for reg_param in np.arange(reg_param_range[0], reg_param_range[1], reg_param_step):
        loss = []
        theta_updated = np.zeros(X_train.shape[1])

        for i in range(iterations_num):
            loss.append(L(theta_updated, reg_param, X_train, y_train))
            theta_updated -= alpha / X_train.shape[0] * grad(theta_updated, reg_param, X_train, y_train)

        cv_loss = L(theta_updated, 0.0, X_cv, y_cv)

        if cv_loss < min_loss:
            min_loss = cv_loss
            min_reg_param = reg_param
            theta = theta_updated

    return min_reg_param, theta

reg_param_range = (0, 1)
reg_param_step = 0.05

min_reg_param, theta = train_model(X_train, y_train, X_cv, y_cv, reg_param_range, reg_param_step)

print(f"Наименьшее значение функции потерь на кросс-валидационном сете достигается при reg_param = {min_reg_param:.2f}")

#точность = (количество правильно классифицированных примеров) / (размер выборки)
y_pred_train_with = np.round(h_theta(theta, X_train))
num_correct = 0
for i in range(len(y_train)):
    if y_pred_train_with[i] == y_train[i]:
        num_correct += 1

accuracy = num_correct / X_train.shape[0]
print('Точность на тренировочной выборке с регуляризацией: ', accuracy)

y_pred_cv_with = np.round(h_theta(theta, X_cv))
num_correct = 0
for i in range(len(y_cv)):
    if y_pred_cv_with[i] == y_cv[i]:
        num_correct += 1

accuracy = num_correct / X_cv.shape[0]
print('Точность на кроссвалидационной выборке с регуляризацией: ', accuracy)

y_pred_test_with = np.round(h_theta(theta, X_test))
num_correct = 0
for i in range(len(y_test)):
    if y_pred_test_with[i] == y_test[i]:
        num_correct += 1

accuracy = num_correct / X_test.shape[0]
print('точность на тестовой выборке с регуляризацией: ', accuracy)
