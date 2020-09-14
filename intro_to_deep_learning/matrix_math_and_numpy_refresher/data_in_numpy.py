import numpy as np


# Scalars
s = np.array(5)
print(s)
print(s.shape)

x = s + 3
print(x)
print(type(x))
print(x.shape)


# Vectors
v = np.array([1, 2, 3])
print(v)
print(v.shape)
x = v[1]
print(x)
print(v[1:])


# Matrices
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(m)
print(m.shape)
print(m[1][2])


# Tensors
# Tensors are just like vectors and matrices, but they can have more dimensions.
# For example, to create a 3x3x2x1 tensor, you could do the following:
t = np.array(
    [
        [
            [[1], [2]],
            [[3], [4]],
            [[5], [6]]
        ],
        [
            [[7], [8]],
            [[9], [10]],
            [[11], [12]]
        ],
        [
            [[13], [14]],
            [[15], [16]],
            [[17], [18]]
        ],
    ]
)
print(t)
print(t.shape)
print(t[2][1][1][0])


# Changing shapes
v = np.array([1, 2, 3, 4])
print(v)
print(v.shape)
x = v.reshape(1, 4)
print(x)
print(x.shape)
x = v.reshape(4, 1)
print(x)
print(x.shape)

x = v[None, :]  # same as v.reshape(1, 4)
print(x)
print(x.shape)
x = v[:, None]  # same as v.reshape(4, 1)
print(x)
print(x.shape)


