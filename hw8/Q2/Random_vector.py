import numpy as np
import matplotlib.pyplot as plt

mean = np.array([2, 1])
var = np.array([[4, 1], [1, 2]])
N = 5000
x0 = np.random.normal(0, 1, N)
x1 = np.random.normal(0, 1, N)
x = np.vstack((x0, x1)).T

L = np.linalg.cholesky(var)
samples = mean + x @ L.T

# Plot the resulting samples as a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of a 2D Gaussian random vector')
plt.show()