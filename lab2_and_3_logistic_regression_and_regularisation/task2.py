import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'ex2data2.txt'
df = pd.read_csv(path, header=None,
                 names=['1st_test', '2nd_test', 'decision_on_correspondence'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df['1st_test'], df['2nd_test'], df['decision_on_correspondence'], s=10)
ax.set_xlabel('Оценка микрочипа по первому тесту', fontsize=8)
ax.set_ylabel('Оценка микрочипа по второму тесту', fontsize=8)
ax.set_zlabel('Решение о соответствии допустимым характеристикам', fontsize=8)
plt.show()


X0 = np.ones((df['1st_test'].size, 3))
X0[:, 1] = df['1st_test'].values
X0[:, 2] = df['2nd_test'].values

# вектор выходных данных
y = np.array(df['decision_on_correspondence']).reshape(-1, 1)

max_degree = 30
X_strike = np.column_stack([X0[:, 1] ** i * X0[:, 2] ** (degree - i) for degree in range(max_degree + 1)
                            for i in range(degree + 1)])

idx = np.array([i for i in range(len(X_strike))])
np.random.shuffle(idx)
split_idx = int(0.8 * len(X_strike))

X_train, X_test = X_strike[idx[:split_idx]], X_strike[idx[split_idx:]]
y_train, y_test = y[idx[:split_idx]], y[idx[split_idx:]]


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
iterations_num = 30000


def grad_regul(theta, lambda_, X, y):
    grad = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        grad += (h_theta(theta, X[i]) - y[i, 0]) * X[i]
    if lambda_ != 0:
        grad += theta
    return grad


loss_1 = []
theta_updated_1 = np.zeros(X_train.shape[1]).T

for i in range(iterations_num):
    loss_1.append(L(theta_updated_1, 0.0, X_train, y_train))
    theta_updated_1 -= alpha / X_train.shape[0] * grad_regul(theta_updated_1, 0.0, X_train, y_train)

plt.plot(loss_1, 'lightskyblue')

loss_2 = []
theta_updated_2 = np.zeros(X_train.shape[1]).T

for i in range(iterations_num):
    loss_2.append(L(theta_updated_2, 1.0, X_train, y_train))
    theta_updated_2 -= alpha / X_train.shape[0] * grad_regul(theta_updated_2, 1.0, X_train, y_train)

plt.plot(loss_2, 'plum')

plt.xlabel('Число итераций')
plt.ylabel('Значения функции потерь')
plt.legend(['lambda = 0.0', 'lambda = 1.0'])
plt.show()


# точность = (количество правильно классифицированных примеров) / (размер выборки)

y_pred_train_without = np.round(h_theta(theta_updated_1, X_train))
num_correct = 0
for i in range(len(y_train)):
    if y_pred_train_without[i] == y_train[i]:
        num_correct += 1

accuracy = num_correct / X_train.shape[0]
print('Точность на тренировочной выборке без регуляризации: ', accuracy)

y_pred_train_with = np.round(h_theta(theta_updated_2, X_train))
num_correct = 0
for i in range(len(y_train)):
    if y_pred_train_with[i] == y_train[i]:
        num_correct += 1

accuracy = num_correct / X_train.shape[0]
print('Точность на тренировочной выборке с регуляризацией: ', accuracy)


y_pred_test_without = np.round(h_theta(theta_updated_1, X_test))
num_correct = 0
for i in range(len(y_test)):
    if y_pred_test_without[i] == y_test[i]:
        num_correct += 1

accuracy = num_correct / X_test.shape[0]
print('Точность на тестовой выборке без регуляризации: ', accuracy)

y_pred_test_with = np.round(h_theta(theta_updated_2, X_test))
num_correct = 0
for i in range(len(y_test)):
    if y_pred_test_with[i] == y_test[i]:
        num_correct += 1

accuracy = num_correct / X_test.shape[0]
print('Точность на тестовой выборке с регуляризацией: ', accuracy)


x_grid, y_grid = np.meshgrid(np.linspace(np.array(df)[:, 0].min(), np.array(df)[:, 0].max(), 500),
                              np.linspace(np.array(df)[:, 1].min(), np.array(df)[:, 1].max(), 500))

gen_grid = np.column_stack([x_grid.ravel() ** i * y_grid.ravel() ** (degree - i)
                            for degree in range(max_degree + 1)
                            for i in range(degree + 1)])

threshold = 0.5
without_reg = (h_theta(theta_updated_1, gen_grid).reshape(500, 500) > 0.5).astype(int)
with_reg = (h_theta(theta_updated_2, gen_grid).reshape(500, 500) > 0.5).astype(int)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))


axs[0].contour(without_reg, levels=[threshold], colors='black', extent=(np.array(df)[:, 0].min(), np.array(df)[:, 0].max(), np.array(df)[:, 1].min(), np.array(df)[:, 1].max()))
colors = ['lightskyblue', 'plum']
axs[0].scatter(np.array(df)[:, 0], np.array(df)[:, 1], c=[colors[i] for i in y.squeeze()], s=10)
axs[0].set_title('Без регуляризации')

axs[1].contour(with_reg, levels=[threshold], colors='black', extent=(np.array(df)[:, 0].min(), np.array(df)[:, 0].max(), np.array(df)[:, 1].min(), np.array(df)[:, 1].max()))
colors = ['lightskyblue', 'plum']
axs[1].scatter(np.array(df)[:, 0], np.array(df)[:, 1], c=[colors[i] for i in y.squeeze()], s=10)
axs[1].set_title('С регуляризацией')

plt.show()
