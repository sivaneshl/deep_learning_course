import numpy as np

# adding a scalar to a vector
values = [1, 2, 3, 4, 5]
values = np.array(values) + 5
print(values)

values = np.array([1, 2, 3, 4, 5])
values += 5
print(values)

# matrix operations
a = np.array([[1, 3], [5, 7]])
b = np.array([[2, 4], [6, 8]])
print (a + b)