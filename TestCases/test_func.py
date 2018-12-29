import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = np.reshape(x, [2, 4])
z = np.expand_dims(y, -1)
print(z.shape)
