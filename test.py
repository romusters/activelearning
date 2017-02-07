probs = [[1.74145531654, -1.74145293236], [0.781296014786, -0.781293451786], [0.881425738335, -0.88142323494],  [0.186622142792, -0.18661954999], [0.374709069729, -0.374706447124]]
n_classes = 2
dict = {}
for i in range(n_classes):
	dict[i] = []

for prob in probs:
	for idx in range(n_classes):
		dict[idx].append(prob[idx])
print dict

import pandas as pd
print pd.DataFrame(dict)
