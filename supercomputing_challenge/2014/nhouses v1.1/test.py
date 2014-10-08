import numpy as np
import matplotlib.pyplot as plt

a = np.zeros((3,4), 'f')
b = np.ones((3,4), 'f')*3.14
c = a + 2*b

plt.plot(c)
plt.show()
