def onehot(x, n_classes):
	import numpy as np
	tmp = np.zeros(n_classes + 1)
	tmp[x] = 1
	return tmp

def cosine_similarity(v1,v2):
	import math
	# compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(len(v1)):
		x = v1[i]; y = v2[i]
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	sim = sumxy/math.sqrt(sumxx*sumyy)
	# sim = sumxy/math.sqrt(sumyy)
	return sim