import numpy as np

# Element wise multiplication
m = np.array([[1, 2, 3], [4, 5, 6]])
n = m * 0.25
print(n)

print(m * n)

print(np.multiply(m, n))

# Matrix product
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(a)
print(a.shape)

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(b)
print(b.shape)

c = np.matmul(a, b)
print(c)
print(c.shape)

# Numpy dot function
a = np.array([[1, 2], [3, 4]])
c = np.dot(a, a)
print(c)
c = a.dot(a)
print(c)
# dot and matmul are the same if the matrices are two dimensional
c = np.matmul(a, a)
print(c)


