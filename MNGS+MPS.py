#Методы наискорейшего градиентного и покоординатного спуска
import scipy.linalg
import numpy as np
import math
from numpy import linalg

np.set_printoptions(linewidth=200)


def matrix(n):
    """Генерируем матрицу А и столбец b"""
    A = np.random.randint(-19, 19, (n, n))
    while not np.all(np.linalg.eigvals(A) > 0):
        A = np.random.rand(n, n)
    b = np.random.randint(-19, 19, (n, 1))
    e = np.random.randint(0, 10, (n, 1))
    return A, b, e


def positivedef(A):
    A=np.dot(A,A.T)
    return A

def Q(A, b, e):
    """ Генерируем значения для вычисления мю
    Возвращаем мюююююююююююю"""
    q = np.dot(A, e) + b
    Aq = np.dot(A, q)
    y = - (np.linalg.norm(q) * np.linalg.norm(q)) / np.dot(q.T, Aq)
    print(y, '    y1')
    # print('y=', y)
    return y, q


def znach(x1, y, q, i):
    """Находим значение следующего приближения"""
    # print('y*q',y * q)
    # x2=[int(a+b) for a,b in zip(x1,y*q)]
    x2 = x1 + y * q
    i += 1
    return x2, i

print("print size of matrix A")
n = int(input())
A, b, e = matrix(n)
#A = positivedef(A)
solve = np.linalg.solve(A, b) # Решение
print(A, '    ', b)
print(Q(A, b, e))
print('SOLVE')
print('eee',e)
x1 = e
i = 0
y, q = Q(A, b, x1)
x2, i = znach(x1, y, q, i)
while np.linalg.norm(x2 - x1) > 1e-6 :   
    x1 = x2                                             # Остановка по норме разницы или большому количеству итераций
    y, q = Q(A, b, x1)
    x2, i = znach(x1, y, q, i)
    print(i, '-', x2)
print('end  ', x2)


E = np.eye(n, dtype=int)
E = np.matrix(E)
x1 = b
i = 0
while True:
    q = E[i % n].T
    y = - (q.T @ (A @ x1 + b) / (q.T @ A @ q))
    x2 = x1
    x1 = x1 + np.ndarray.item(y) * q
    i += 1
    if np.linalg.norm(x1 - x2) < 1e-6:
        break
x2 = x1
i_2 = i
print()
print(x2,'    ',i_2,'mps')
print()
print('SOLVE', solve)

