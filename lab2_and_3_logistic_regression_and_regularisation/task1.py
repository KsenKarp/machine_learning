import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex2data1.txt'
df = pd.read_csv(path, header=None, names=['scores_1st_exam', 'scores_2nd_exam', 'decision_on_enrollment'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df['scores_1st_exam'], df['scores_2nd_exam'], df['decision_on_enrollment'], s=10)
ax.set_xlabel('Баллы за первый экзамен', fontsize=8)
ax.set_ylabel('Баллы за второй экзамен', fontsize=8)
ax.set_zlabel('Решение о зачислении', fontsize=8)

plt.show()


# Нормализуем данные

a1 = df['scores_1st_exam'].sum() / df['scores_1st_exam'].count()
a2 = df['scores_2nd_exam'].sum() / df['scores_2nd_exam'].count()

disp1 = np.sqrt(((df['scores_1st_exam'] - a1) ** 2).sum() / df['scores_1st_exam'].count())
disp2 = np.sqrt(((df['scores_2nd_exam'] - a2) ** 2).sum() / df['scores_2nd_exam'].count())


df['scores_1st_exam'] = (df['scores_1st_exam'] - a1) / disp1
df['scores_2nd_exam'] = (df['scores_2nd_exam'] - a2) / disp2


X0 = np.ones((df['scores_1st_exam'].size, 3))
X0[:, 1] = df['scores_1st_exam'].values
X0[:, 2] = df['scores_2nd_exam'].values

theta = np.array([0, 0, 0]).T
y = df['decision_on_enrollment']

# Oбъем выборки
m = y.size


def g(z):
    return 1 / (1 + np.exp(-z))


def h_theta(theta, X0):
    return g(np.dot(X0, theta))


# Функция потерь
def L(theta):
    return -np.sum(y * np.log(h_theta(theta, X0)) + (1 - y) * np.log(1 - h_theta(theta, X0))) / m


alpha = 0.01                 # Скорость обучения
alpha1 = 0.5
iterations_num = 20000


def grad_descent(theta, alpha):
    loss = []
    for i in range(iterations_num):
        loss.append(L(theta))
        theta = theta - (alpha / m) * np.dot((h_theta(theta, X0) - y).T, X0).T
    return theta, loss


theta_new, loss = grad_descent(theta, alpha)
theta_new1, loss1 = grad_descent(theta, alpha1)

# Проверка сходимости алгоритма обучения
plt.plot(loss)
plt.xlabel('Число итераций')
plt.ylabel('Значение функции потерь')

plt.plot(loss1)
plt.xlabel('Число итераций')
plt.ylabel('Значение функции потерь')
plt.legend(['Функция потерь при alpha = 0.01', 'Функция потерь при alpha = 0.5'])
plt.show()

# Построение границы принятия решения
plt.plot(X0[:, 1], -(theta_new[0] + X0[:, 1] * theta_new[1]) / theta_new[2], 'crimson')

colors = ['plum', 'lightskyblue']
plt.scatter(df['scores_1st_exam'], df['scores_2nd_exam'], c=[colors[i] for i in y.squeeze()], s=10)

plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


print('Параметры, полученные  с помощью алгоритма градиентного спуска: ', theta_new)
print('Вероятность того, что абитуриент, набравший 45 и 85 за экзамены, поступит: ',
      h_theta(theta_new, np.array((1, (45 - a1) / disp1, (85 - a2) / disp2))))
print('')

# точность = (количество правильно классифицированных примеров) / (размер обучающей выборки)
y_pred = np.round(h_theta(theta_new, X0))
num_correct = np.sum(y_pred == y)

accuracy = num_correct / m
print('Точность классификации: ', accuracy)
