import numpy as np

a = [[1, 2, 3, 2], [1, 2, 3, 1], [2, 3, 4, 1], [1, 0, 2, 0], [2, 1, 2, 0], [2, 1, 1, 1]]
b = [[1, 4, 3, 2], [1, 2, 3, 1], [2, 3, 4, 1], [1, 0, 2, 0], [2, 1, 2, 0], [2, 1, 1, 4]]
c = []
c.append(a)
c.append(b)
c = np.array(c)
print(c)
c = np.sum(c, axis=0)
print(c)
print(c.max())
print(a + b)

a = np.array(a)
b = np.array(b)
d = np.concatenate((a, b), axis=1)
print(np.sum([a,b], axis=0))
