import numpy as np
import matplotlib.pyplot as plt
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 1000)
plt.hist(s) 

plt.show()