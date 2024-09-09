import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

data.insert(0, 'Ones', 1)

cols = data.shape[1]

X = data.iloc[:, 0:cols-1] #[row_idx, column_idx]
y = data['Profit']

alpha = 0.01
iterations = 10000


def predCost(x, weights):
    return np.dot(x, weights)


def computeCost(X, y, w):  #функция потерь
    return np.sum((predCost(X, w) - y) ** 2) / (2 * len(data['Profit']))


theta = np.array([0, 0]).T


def gradientDescent(X, y, theta, alpha, iters):
    cost = []
    for i in range(iters):
        theta = theta - alpha / (len(y)) * np.dot((predCost(X, theta) - y).T, X).T
        cost.append(computeCost(X, y, theta))
    return theta, cost  # возвращает конечные тэта и массив ошибок от количества итераций


t_final, cost_final = gradientDescent(X, y, theta, alpha, iterations)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = t_final[0] + (t_final[1] * x)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, f, c='g', label='Предсказание')
ax.scatter(data.Population, data.Profit, label='Тренировочные данные')
ax.legend(loc=2)
ax.set_xlabel('Население')
ax.set_ylabel('Прибыль')
ax.set_title('Предсказанная прибыль от населения города')
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iterations), cost_final, 'r')
ax.set_xlabel('Количество итераций')
ax.set_ylabel('Цена')
plt.show()


x1 = np.matrix([1, 5])
x2 = np.matrix([1, 10])
print("Cпрогнозированная цена для города с населением 5 тыс. человек методом градиентного спуска:", predCost(x1, t_final))
print("Cпрогнозированная цена для города с населением 10 тыс. человек методом градиентного спуска:", predCost(x2, t_final))

t_normal = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

print("Cпрогнозированная цена для города с населением 5 тыс. человек методом нормального ур-ия:", predCost(x1, t_normal))
print("Cпрогнозированная цена для города с населением 10 тыс. человек методом нормального ур-ия:", predCost(x2, t_normal))

print()
print('Параметры, найденные методом градиентного спуска:', t_final.T)
print('Параметры, найденные методом нормального ур-ия:', t_normal)
