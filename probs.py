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

filter = False # filter subject tweets
def probs_ntokens(filter, n_tokens):
    for token in n_tokens:
        data = pd.read_hdf(root + "results/test/ntokens/hashtag/" + str(token) + "/0/probs.h5")

        df = pd.merge(data, dset.tweets, on="id")

        # df = pd.DataFramxe({"id": ids, "probs": probs, "filtered_text": dset.tweets.filtered_text, "text": dset.tweets.text})
        df = df.sort_values(by=[0], ascending=False)  # TEST!
        print "Data size before deduplification: %i" % df.id.count()
        df = df.drop_duplicates("id")
        print "Data size after deduplification: %i" % df.id.count()

        print "Data size before nan drop: %i " % df.id.count()
        df = df.dropna()
        print "Data size after nan drop: %i" % df.id.count()

        rm_list = ["<stopword>", "<mention>", "<url>", "rt"]
        df["ntokens"] = df.filtered_text.apply(lambda x: len([a for a in x.split() if a not in rm_list]))
        df = df[df.ntokens == token]
        if filter:
            df = df[~df.filtered_text.str.contains("voetbal")]
        print df
        df.to_csv(root + "results/" + str(token) + ".csv")


def probs(filter):
    data = pd.read_hdf(root + "results/test/0/probs.h5")
    df = pd.merge(data, dset.tweets, on="id")
    df = df.sort_values(by=[0], ascending=False)
    print "Data size before deduplification: %i" % df.id.count()
    df = df.drop_duplicates("id")
    print "Data size after deduplification: %i" % df.id.count()

    print "Data size before nan drop: %i " % df.id.count()
    df = df.dropna()
    print "Data size after nan drop: %i" % df.id.count()

    rm_list = ["<stopword>", "<mention>", "<url>", "rt"]
    df["ntokens"] = df.filtered_text.apply(lambda x: len([a for a in x.split() if a not in rm_list]))

    if filter:
        df = df[~df.filtered_text.str.contains("voetbal")]
    print df
    df.to_csv(root + "results/all_tokens.csv")

# probs(filter)
# probs_ntokens(False, [5])
# for n_token in n_tokens:
#     probs(filter, n_token)