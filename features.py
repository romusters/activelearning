def onehot(x, n_classes):
	import numpy as np
	tmp = np.zeros(n_classes + 1)
	tmp[x] = 1
	return tmp