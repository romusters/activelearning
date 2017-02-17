import dataset
dset = dataset.Dataset()
nn_data = dset.get_seed_dataset([0,4])
nn_data["labels"] = nn_data["labels"].replace(4, 1)
nn_data = dset.transform_dataset(nn_data, 2)

import numpy as np
import features

# trainset = nn_data.sample(frac=0.8, random_state=200)
# testset = nn_data.drop(trainset.index)
#
import pandas as pd
# n_classes = 2
# result = {}
# result["trainset"] = np.array(trainset[range(70)].values.tolist())
# result["trainlabels"] = np.array(trainset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
# result["trainids"] = np.array(trainset.id.values.tolist())
# result["testdata"] = np.array(testset[range(70)].values.tolist())
# result["testlabels"] = np.array(testset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
# result["testids"] = np.array(testset.id.values.tolist())
# result["nclasses"] = 1
# result["allvectors"] = np.array(dset.all_vectors_store["data"][range(70)].values.tolist())
# result["allvectorsids"] = np.array(dset.all_vectors_store["data"]["id"].values.tolist())


import nn
path = "/media/cluster/data1/lambert/results/test/"
nn.train_nn(path, nn_data, 0)

# # vergelijk jihad2.csv met een nn die 1v1 doet
# import pandas as pd
# data = pd.read_hdf("/media/cluster/data1/lambert/results/seeds/0/probs.h5")
# probs = data[1]
# ids = data.id
# df = pd.DataFrame({"id": ids, "probs": probs, "text": dset.tweets.text})
# df = df.sort_values(by=["probs"], ascending=False) #TEST!
# df.to_csv("/media/cluster/data1/lambert/results/probs/jihad3.csv")
