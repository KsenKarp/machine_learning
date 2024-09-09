import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data.head()

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.set_xlabel('Размер дома в квадратных футах', fontsize=8)
ax.set_ylabel('Количество спален', fontsize=8)
ax.set_zlabel('Цена дома', fontsize=8)
ax.scatter3D(data['Size'], data['Bedrooms'], data['Price'])


avg_size = data['Size'].sum() / data['Size'].count()
avg_bedr = data['Bedrooms'].sum() / data['Bedrooms'].count()
disp_size = np.sqrt(((data['Size'] - avg_size) ** 2).sum() / data['Size'].count())
disp_bedr = np.sqrt(((data['Bedrooms'] - avg_bedr) ** 2).sum() / data['Bedrooms'].count())


data['Size'] = (data['Size'] - avg_size) / disp_size
data['Bedrooms'] = (data['Bedrooms'] - avg_bedr) / disp_bedr
data.head()


data.insert(0, 'Ones', 1)

cols = data.shape[1]

X = data.iloc[:, 0:cols-1] #[row_idx, column_idx]
y = data['Price']


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.set_xlabel('Отнормированная шкала размера дома', fontsize=8)
ax.set_ylabel('Отнормированная шкала количества спален', fontsize=8)
ax.set_zlabel('Цена дома', fontsize=8)
ax.scatter3D(data['Size'], data['Bedrooms'], data['Price'])
plt.show()

alpha = 0.01
iterations = 10000


def predCost(x, weights):
    return np.dot(x, weights)


def computeCost(X, y, w):  #функция потерь
    return np.sum((predCost(X, w) - y) ** 2) / (2 * len(data['Price']))


theta = np.array([0, 0, 0]).T


def gradientDescent(X, y, theta, alpha, iters):
    cost = []
    for i in range(iters):
        theta = theta - alpha / (len(y)) * np.dot((predCost(X, theta) - y).T, X).T
        cost.append(computeCost(X, y, theta))
    return theta, cost  #возвращает конечные тэта и массив ошибок от количества итераций


t_final, cost_final = gradientDescent(X, y, theta, alpha, iterations)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iterations), cost_final, 'r')
ax.set_xlabel('Количество итераций')
ax.set_ylabel('Значение функции потерь')
plt.show()


x1 = np.array((1, (1500 - avg_size) / disp_size, (3 - avg_bedr) / disp_bedr))
print("Cпрогнозированная цена дома методом градиентного спуска:", predCost(x1, t_final))

x_em = np.empty((len(data), 3))
x_em[:, 1] = data['Size']
x_em[:, 2] = data['Bedrooms']
x_em[:, 0] = np.ones(len(data))
t_normal = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


print("Cпрогнозированная цена цена дома методом нормального ур-ия:", predCost(x1, t_normal))
print()
print('Параметры, найденные методом градиентного спуска:', t_final.ravel())
print('Параметры, найденные методом нормального ур-ия:', t_normal)
