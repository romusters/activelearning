import config
import dataset

import pandas as pd
import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import config
# conf = config.get_config()
# root = eval(conf.get("paths", "root"))
# print root

def get_tweet_belonging_to_id():
    id = 2602750216703



    dset = dataset.Dataset(root)
    print dset.tweets[dset.tweets["id"] == id].text.values

# def most_similar_tokens():
#     path = "/home/robert/lambert/models/w2vmodel.h5"
#     print path
#     model = pd.read_hdf(path, "data")
#     voetbal_vector = model[model.words == "voetbal"].vectors
#     model["cossim"] = model.vectors.apply(lambda x: cos)


# get_tweet_belonging_to_id()
# most_similar_tokens()


