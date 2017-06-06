import numpy as np
uniform = np.random.uniform((10,10,100))

mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance
normal = np.random.multivariate_normal(mean, cov, 100)
from scipy.stats import kurtosis,skew
# print kurtosis(uniform)
# print kurtosis(normal)
print skew(uniform)
print skew(normal)