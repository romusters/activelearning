import sys
import logging
import w2v
import pandas as pd
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import config
root, data_path, model_path, vector_path = config.get_paths()

# make seed datasets per hashtag // located in /datasets/subjects

# convert w2v model to hdf5
# w2v.w2vmodel_to_hdf(model_path)
# w2v.test_model(model_path)

# # convert data_jan.csv into sample, bceause else the dataset is just too large
# data = pd.read_csv(data_path + "data.csv", names = ["text", "filtered_text", "id"])
# data = data.dropna()
# data.index = data.id
# data = data.drop("id", axis=1)
# # print data
# sample = data.sample(frac=0.1)
# sample.to_csv(data_path + "data_sample.csv")


# convert data to vectors with id
# w2v.get_vectors(data_path + "data_sample.csv", vector_path, model_path + "w2vmodel")



# First iteration, train seeds: voetbal versus moslim, because in noise there could also be tweets about the subject
import dataset
dset = dataset.Dataset(root)


# get the subject seeds and ..
dset.create_subject_sets()
# import sys
# sys.exit(0)

#.. balance them.
# dset.balance_data()
import sys
sys.exit(0)



# train seeds versus seeds
import nn
for i in range(0, len(dset.all_hashtags)-1, 2):
    logger.info("Getting seeds for subject pair: %s, %s", dset.all_hashtags[i], dset.all_hashtags[i+1])
    nn_data = dset.get_seed_dataset([i, i+1])
    logger.info("Transforming data.")
    nn_data = dset.transform_dataset(nn_data, i,  2)

    nn.train_nn(root + "results/seeds/", nn_data, i)
    dset.make_probs_file(dset.all_hashtags[i], i, 0)
    dset.make_probs_file(dset.all_hashtags[i+1], i, 1)
    import sys
    sys.exit(0)



# # Get the thresholds belonging to the first two seeds
import al
for i in range(4, len(dset.all_hashtags)):
    print dset.all_hashtags[i]
    al.find_threshold_subject(dset.all_hashtags[i], root)

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
    if i < 0:
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
