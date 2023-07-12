import numpy as np
import matplotlib.pyplot as plt

N = 5000
w0 = np.random.normal(0, 1, N)
w1 = np.random.normal(0, 1, N)
w = np.vstack((w0, w1)).T

plt.figure(figsize=(6, 6))
plt.scatter(w[:, 0], w[:, 1], s=1)
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('Scatter plot of a 2D white random vector')
plt.show()

A = np.array([[-1, -1], [2, 4]])
w_trans = (A@w.T).T
y_cov = np.cov(w_trans.T)
print("y cov_matrix:", y_cov)

plt.figure(figsize=(6, 6))
plt.scatter(w_trans[:, 0], w_trans[:, 1], s=1)
plt.xlabel('y0')
plt.ylabel('y1')
plt.title('Scatter plot of a 2D random vector after transformation')
plt.show()

_, ei_vector = np.linalg.eig(y_cov)
v = (ei_vector.T @ w_trans.T).T
v_cov = np.cov(v.T)
print("v cov_matrix:", v_cov)

plt.figure(figsize=(6, 6))
plt.scatter(v[:, 0], v[:, 1], s=1)
plt.xlabel('v0')
plt.ylabel('v1')
plt.title('Scatter plot of a 2D random vector after whittening y')
plt.show()

ei_value_v, _ = np.linalg.eig(y_cov)
ei = np.zeros((ei_value_v.shape[0], ei_value_v.shape[0]))
for i in range(ei_value_v.shape[0]):
    ei[i, i] = 1 / np.sqrt(ei_value_v[i])

z = (ei @ v.T).T
z_cov = np.cov(z.T)
print("z cov_matrix:", z_cov)

plt.figure(figsize=(6, 6))
plt.scatter(z[:, 0], z[:, 1], s=1)
plt.xlabel('z0')
plt.ylabel('z1')
plt.title('Scatter plot of a 2D random vector of z')
plt.show()