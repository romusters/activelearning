import config
import dataset

import pandas as pd
import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import config
conf = config.get_config()
root = eval(conf.get("paths", "root"))
print root

def get_tweet_belonging_to_id():
    id = 2602750216703



    dset = dataset.Dataset(root)
    print dset.tweets[dset.tweets["id"] == id].text.values

get_tweet_belonging_to_id()