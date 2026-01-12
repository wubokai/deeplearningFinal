a = []
import numpy as np

l1 = np.random.randn( 5, 82, 2)
l2 = np.reshape(l1,(5, 82, 2, 1))
print(l2.shape)
