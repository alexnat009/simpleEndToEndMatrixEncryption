import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, linsolve, lambdify
from matplotlib import cm


def z_func(x1, x2):
    return 26 * (x1 ** 2) + 30 * x1 * x2 + 19 * (x2 ** 2) - x1 - x2


A = np.array([[1, 5], [5, 2]])
b = np.array([1, 1])
A = np.dot(A, A.T)

X1 = np.linspace(-4, 3, 50)
X2 = np.linspace(-4, 3, 50)
X1, X2 = np.meshgrid(X1, X2)
Z = np.clip(z_func(X1, X2), 0, 1000)

x, y = symbols("x y", real=True)
z = z_func(x, y)

x_min, y_min = list(linsolve([diff(z, x), diff(z, y)], (x, y)))[0]
Z_min = np.min(Z)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_title('Graph of quadratic Form')
ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')
ax.set_zlabel('$Z$')
surf = ax.plot_surface(X1, X2, Z, rstride=2, cstride=2, cmap=cm.hot, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.scatter(Z_min, x_min, y_min, marker='H', color='blue', s=100)

grad_fun = [diff(z, var) for var in (x, y)]
num_grad_fun = lambdify([x, y], grad_fun)
grad_dat = num_grad_fun(X1, X2)
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('Graph of gradients of quadratic Form')
ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')
ax.set_zlabel('$Z$')

ax.quiver(X1, X2, Z, grad_dat[0], grad_dat[1], Z - 1, length=0.001)

plt.show()

