# test if the amount of tokens has an influence on the nn performance
# we saw that it doesnt work well when all tweet of varying amount of tokens is used.

import pandas as pd
import numpy as np
import os, sys
import features
import config
root, data_path, model_path, vector_path = config.get_paths()
print root

import pandas as pd






import dataset
dset = dataset.Dataset(root)
# dset.create_subject_sets()


## Train a neural network on amount of tokens individually
vectors = dset.all_vectors_store["data"]
voetbal = pd.read_csv(root + "datasets/seeds/voetbal.csv")
voetbal = pd.merge(voetbal, vectors, on="id")
voetbal["labels"] = 0
jihad = pd.read_csv(root + "datasets/seeds/jihad.csv")
jihad = pd.merge(jihad, vectors, on="id")
jihad["labels"] = 1



#determine tokens
rm_list = ["<stopword>", "<mention>", "<url>", "rt"]
voetbal["ntokens"] = voetbal.filtered_text.apply(lambda x: len([a for a in x.split() if a not in rm_list]))
jihad["ntokens"] = jihad.filtered_text.apply(lambda x: len([a for a in x.split() if a not in rm_list]))

# make data batch small enough 12:22-

#determine largest group tokens->10
# print merged.groupby("ntokens").agg("count")
for i in range(1, 19, 1):
    print "Ntokens == %i" % i
    #get vectors from voetbal with i tokens
    d_voetbal = None
    try:
        d_voetbal = voetbal[voetbal.ntokens == i].sample(n=3000, replace=True)
    except ValueError:
        print "voetbal has zero samples"
        continue
    count = len(d_voetbal.index)
    print "Voetbal data size %i" % count
    d_jihad = jihad[jihad.ntokens == i]
    print "Jihad data size pre balance %i" % len(d_jihad.index)

    try:
        d_jihad = d_jihad.sample(n=len(d_voetbal.index), replace=True)
    except ZeroDivisionError:
        print "empty dataset"
    print "Jihad data size post balance %i" % len(d_jihad.index)
    assert(len(d_jihad.index) == len(d_voetbal.index))


    all = d_voetbal.append(d_jihad).sample(frac=1)
    all = all.reset_index()
    print all
    trainset = all.sample(frac=0.8, random_state=200)
    testset = all.drop(trainset.index)
    result = {}
    n_classes = 2
    result["trainset"] = np.array(trainset[range(70)].values.tolist())
    result["trainlabels"] = np.array(trainset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
    result["trainids"] = np.array(trainset.id.values.tolist())
    result["testdata"] = np.array(testset[range(70)].values.tolist())
    result["testlabels"] = np.array(testset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
    result["testids"] = np.array(testset.id.values.tolist())
    result["nclasses"] = 2
    tmp_data = dset.all_vectors_store["data"]#.sample(n=43000)
    result["allvectors"] = np.array(tmp_data[range(70)].values.tolist())
    result["allvectorsids"] = np.array(tmp_data["id"].values.tolist())
    print result
    import nn
    path = root + "results/test/ntokens/hashtag/"
    os.mkdir(path + str(i))
    nn.train_nn(path + str(i) +"/", result, 0)



# make validation set, not containing ids from data_sample.