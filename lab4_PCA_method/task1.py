import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig

path = 'ex_pca_data1.csv'
df = pd.read_csv(path, header=None,
                 names=['Первый признак', 'Второй признак'])

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(df['Первый признак'], df['Второй признак'], c='blue', s=10)
ax.set_xlabel('Первый признак', fontsize=8)
ax.set_ylabel('Второй признак', fontsize=8)

plt.show()

#нормализация

a1 = df['Первый признак'].sum() / df['Первый признак'].count()
a2 = df['Второй признак'].sum() / df['Второй признак'].count()

σ1 = np.sqrt(((df['Первый признак'] - a1) ** 2).sum() / df['Первый признак'].count())
σ2 = np.sqrt(((df['Второй признак'] - a2) ** 2).sum() / df['Второй признак'].count())


df['Первый признак'] = (df['Первый признак'] - a1) / σ1
df['Второй признак'] = (df['Второй признак'] - a2) / σ2


m = df['Первый признак'].size

# корреляционная матрица
Sigma = 1 / m * np.dot(df.T, df)

eigenvalues, eigenvectors = eig(Sigma)

# находим индексы собственных значений в порядке убывания
sorted_indices = np.argsort(eigenvalues)[::-1]

# меняем местами
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print('Главные компоненты:')
print(eigenvectors)


print('Выборочные дисперсии данных на эти направления:')
print(eigenvalues.real)


print('Доля сохраненной дисперсии на первую главную компоненту:')
print((eigenvalues[0] / (eigenvalues.sum())).real)

k = 1
C1 = eigenvectors[:, :k]


projections = np.dot(df, C1)
plt.scatter(projections, np.zeros_like(projections), color='lightskyblue')
plt.axhline(y=0, linestyle='-', linewidth=1, color='lightskyblue')
plt.show()

reconstructed_data = np.dot(projections, C1.T)

print('Исходные и восстановленные признаки 1 элемента выборки:')
print(np.array(df.iloc[0]), reconstructed_data[0])


fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot()
ax.scatter(df['Первый признак'], df['Второй признак'], c='blue', s=10)


x = reconstructed_data[:, 0]
y = reconstructed_data[:, 1]

ax.scatter(x, y, c='red', s=10, marker='x')
ax.plot(x, y, c='red', linewidth=1)


ax.set_xlabel('Первый признак', fontsize=8)
ax.set_ylabel('Второй признак', fontsize=8)

for i in range(len(x)):
    ax.plot([df['Первый признак'][i], x[i]], [df['Второй признак'][i], y[i]], c='blue', linewidth=1)

plt.show()
