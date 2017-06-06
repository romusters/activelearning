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
# make seed datasets per hashtag // located in /datasets/subjects


# First iteration, train seeds: voetbal versus moslim, because in noise there could also be tweets about the subject
import dataset
dset = dataset.Dataset(root)

# nn_data = dset.get_seed_dataset([0,1])
# nn_data = dset.transform_dataset(nn_data, 2)
#
# import nn
#
# nn.train_nn(root + "results/seeds/", nn_data, 0)

#
#
# # Get the thresholds belonging to the first two seeds
import al
# dset.make_probs_file("voetbal", 0)
# al.find_threshold_subject( "voetbal", root)

dset.make_probs_file("moslim", 0, 1)
al.find_threshold_subject("moslim", root)
import sys
sys.exit(0)

# get the threshold
# t_00 = al.get_threshold_subject(root + "results/thresholds/", "voetbal")
# t_01 = al.get_threshold_subject(root + "results/thresholds/", "moslim")
#
#
# # Use the threshold to get more polished datasets using the probabilities
# dset.save_meta_dataset( "datasets/meta_subjects/", dset.all_vectors_store, "voetbal", 0, t_00)
# dset.save_meta_dataset( "datasets/meta_subjects/", dset.all_vectors_store, "moslim", 1, t_01)

# Now train the network with the more polished datasets
# The first step is to train the first subject versus the rest...
# meta_dataset_voetbal = dset.get_meta_dataset(["voetbal"])
# dataset = dset.transform_dataset_meta(meta_dataset_voetbal, 2)
# nn.train_nn("results/meta_subjects/", dataset, 0)

# ...then train the first, second subject with the rest.
# sub_dataset = dset.get_meta_dataset(["voetbal", "moslim"])
# dataset = dset.transform_dataset(sub_dataset, 3)
#
# # Train metadataset for 2 subjects versus the rest
# nn.train_nn("results/meta_subjects/", dataset, 1)


for i, hashtag in enumerate(dset.all_hashtags):
	if i < 4:
		continue
	# Get the previous meta_subjects and add the seed
	seed_hashtag = dset.all_hashtags[i]
	meta_hashtags = dset.all_hashtags[0:i]
	logger.info("The meta hashtags are: %s ", meta_hashtags)
	print seed_hashtag
	seed_index = dset.all_hashtags.index(seed_hashtag)
	print seed_index
	sub_dataset = dset.get_meta_seed_dataset(meta_hashtags, [seed_index])

	dataset = dset.transform_dataset(sub_dataset, i + 1)


	# Train a neural network to get the probabilities of the seed dataset
	# previous metasubjects versus the next seed
	logger.info("Training the neural network with the seed: %s", seed_hashtag)
	nn.train_nn(root + "results/seeds/", dataset, i)

	# Load the probabilities and generate the metasubject dataset for the current seed, it contains: id, probs, text
	dset.make_probs_file(hashtag, i)

	# Now we can find the threshold for the metasubject dataset for the last seed
	# al.find_threshold_subject(seed_hashtag)

	# Generate the metasubject dataset for the last seed using the threshold
	# seed_hashtag = "werk"
	threshold = al.get_threshold_subject(root + "results/thresholds/", seed_hashtag)
	#
	dset.save_meta_dataset(seed_hashtag, threshold, i)
	#
	# # Train the network using the new metadataset and the previous metadatasets
	sub_hashtags = dset.all_hashtags[0:i+1]
	print sub_hashtags
	nn_data = dset.get_meta_dataset(sub_hashtags)
	nn_data = dset.transform_dataset(nn_data, i + 2)
	print nn_data
	logger.info("Training the neural network with the metasubject %s", meta_hashtags)
	nn.train_nn(root + "results/meta_subjects/", nn_data, i)



	# Use previous weights
	# nn.train_nn(root + "results/", nn_data, i)

	# Remove duplicates
