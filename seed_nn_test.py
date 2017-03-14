import pandas as pd
import os
import sys
import logging
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import config
root, data_path, model_path, vector_path = config.get_paths()
print root


new_data = False
path = root + "results/test/0/two_class.npy"
if False:
    import dataset
    dset = dataset.Dataset(root)
    nn_data = dset.get_seed_dataset([0,4]) # voetbal and jihad
    nn_data["labels"] = nn_data["labels"].replace(4, 1)
    nn_data = dset.transform_dataset(nn_data, 0, 2)

    logger.info("Saving two class")

    np.save(path, nn_data)
else:
    nn_data = np.load(path).item()
data = pd.DataFrame(nn_data["trainset"])

import numpy as np
import features
#
# trainset = data.sample(frac=0.8, random_state=200)
# testset = data.drop(trainset.index)
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
path = root + "results/test/"
nn.train_nn(path, nn_data, 0)



# vergelijk jihad2.csv met een nn die 1v1 doet
import pandas as pd
data = pd.read_hdf(root + "results/test/0/probs.h5")
tweets = pd.read_csv(root + "datasets/data_sample.csv" )
merged = pd.merge(data, tweets, on="id").sort_values(1)
ntokens = merged
rm_list = ["<stopword>", "<mention>", "<url>", "rt"]

ntokens["count"] = merged.filtered_text.apply(lambda x: len([a for a in x.split() if a not in rm_list]))
filter = ntokens[ntokens["count"] > 10]

df = pd.DataFrame({"id": filter.id, "probs": filter[0], "text": filter.text, "tokens": filter.filtered_text})
df.to_csv(root + "results/test/voetbal.csv")
print df
# probs = data[1]
# print data
# ids = data.id
# df = pd.DataFrame({"id": ids, "probs": probs, "text": dset.tweets.text})
# df = df.sort_values(by=["probs"], ascending=False) #TEST!
# df.to_csv(root + "results/probs/jihad3.csv")


import al
# al.find_threshold_subject("jihad3", root)
# import dataset
# dset = dataset.Dataset(root)
# import pandas as pd
# data = pd.read_hdf(root + "results/test/0/probs.h5")
# probs = data[1]
# ids = data.id
# df = pd.DataFrame({"id": ids, "probs": probs, "text": dset.tweets.text})
# df = df.sort_values(by=["probs"], ascending=False) #TEST!
# df.to_csv(root + "results/test/0/jihad.csv")

# al.find_threshold_subject("voetbal_test", root)