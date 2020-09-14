import numpy as np

m = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(m)
print(m.T)

m_t = m.T
m_t[3][1] = 200
print(m_t)


inputs = np.array([[-0.27,  0.45,  0.64, 0.31]])
print(inputs)
print(inputs.shape)
weights = np.array([[0.02, 0.001, -0.03, 0.036], [0.04, -0.003, 0.025, 0.009], [0.012, -0.045, 0.28, -0.067]])
print(weights)
print(weights.shape)

result = np.matmul(inputs, weights.T)
print(result)

result = np.matmul(weights, inputs.T)
print(result)

