import pandas as pd
import numpy as np
import os, sys
import features
import config
root, data_path, model_path, vector_path = config.get_paths()
print root

import pandas as pd
import dataset
# dset = dataset.Dataset(root)

def make_validate(hashtag):
    f = open(root + "results/thresholds/" + hashtag, "r")
    threshold = float(f.readlines()[0])
    print float(threshold)

    data = pd.read_csv(root + "results/" + hashtag + ".csv", usecols=["id", "0", "text"])
    data = data[data["0"]> threshold]
    data["label"] = 1
    print data.sample(n=100).to_csv(root + "results/thresholds/" + hashtag + "_annotate.csv")

make_validate("10")
make_validate("all_tokens") 