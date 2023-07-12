import numpy as np
import matplotlib.pyplot as plt

#q1
width = 1000
mean1 = np.array([2, 1])
mean2 = np.array([-2, 1])
var1 = np.array([[1, -1], [-1, 4]])
var2 = np.array([[4, 0], [0, 1]])
x = np.arange(-8, 8, (8-(-8))/width)
mesh = np.meshgrid(x, x)
mesh = np.array([mesh[0], mesh[1]])
dis1 = np.zeros((width, width))
dis2 = np.zeros((width, width))
for i in range(width):
    for j in range(width):
        vec = np.array([x[i], x[j]])
        #print("vec", vec)
        #print("var1", np.linalg.inv(var1))
        #print("dot", (vec - mean1).dot(np.linalg.inv(var1)))
        dis1[i, j] = (vec - mean1) @ (np.linalg.inv(var1)) @ (vec-mean1)
        dis2[i, j] = (vec - mean2) @ (np.linalg.inv(var2)) @ (vec-mean2)

B = np.array([0.5, 1, 1.5, 2])
cs = plt.contourf(x, x, dis1.T, levels=B**2,
    colors=['#808080', '#A0A0A0', '#C0C0C0', 'y'], extend='min', alpha=0.5)
cs2 = plt.contourf(x, x, dis2.T, levels=B**2,
    colors=['#808080', '#A0A0A0', '#C0C0C0', 'y'], extend='min', alpha=0.5)
cl = plt.colorbar(cs)
plt.plot(mean1[0], mean1[1], 'o', color='b', label="mean1")
plt.plot(mean2[0], mean2[1], '*', color='g', label="mean2")
plt.xlabel("x0")
plt.ylabel("x1")
plt.title(f"Mahalanobis distance")
plt.legend()
plt.show()

PS1 = 0.1
PS2 = 0.9
one = np.ones((width, width))
dec = -0.5 * np.log(np.linalg.det(var1)) * one - 0.5 * dis1 + np.log(PS1) * one + 0.5 * np.log(np.linalg.det(var2)) * one + 0.5 * dis2 - np.log(PS2) * one
cs = plt.contour(x, x, dec.T, levels=[0], extend='min', alpha=0.5)
plt.plot(mean1[0], mean1[1], 'o', color='b', label="mean1")
plt.plot(mean2[0], mean2[1], '*', color='g', label="mean2")
plt.xlabel("x0")
plt.ylabel("x1")
plt.title(f"Decision boundary for PS1={PS1}, PS2={PS2}")
plt.legend()
plt.show()