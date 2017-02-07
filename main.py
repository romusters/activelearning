import pandas as pd


# make seed datasets per hashtag // located in /datasets/subjects


# First iteration, train seeds: voetbal versus moslim
import dataset
dset = dataset.Dataset()
nn_data = dset.get_dataset([0,1])

import nn
path = "/media/cluster/data1/lambert/results/seeds/"
nn.train_nn(path, nn_data, 0)


# Get the thresholds belonging to the first two seeds
import al
al.find_threshold_subject("voetbal")
al.find_threshold_subject("moslim")


# get the threshold
t_00 = al.get_threshold_subject("voetbal")
t_01 = al.get_threshold_subject("moslim")


# Use the threshold to get more polished datasets using the probabilities
al.save_threshold_subject( "/media/cluster/data1/lambert/meta_subjects/", dset.all_vectors_store, "voetbal", 0, t_00)
al.save_threshold_subject( "/media/cluster/data1/lambert/meta_subjects/", dset.all_vectors_store, "moslim", 1, t_01)


for i, hashtag in enumerate(dset.all_hashtags[2:]):
	# Get the previous meta_subjects and add the seed
	nn_data = dataset.get_meta_seed(dset.all_vectors_store, dset.balanced_store, [0,1], [i])
	nn.train_nn("/media/cluster/data1/lambert/results/", nn_data, i)
