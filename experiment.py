import config
root, data_path, model_path, vector_path = config.get_paths()
print root
import pandas as pd
# run a neural network with and without ntokens selection DONE
# import ntokens_nn_test
# import nn_test

# run a neural without ntokens selection DONE
n_tokens = range(1, 19, 1)
# per ntokens file

# # prepare and let the user annotate first 20 tweets per ntokens file for ..
# import probs
# # .. without subject in tweet
# probs.probs_ntokens(True, n_tokens)
# # make the files which need to be annotated per user
# import prepare_experiment
# prepare_experiment.probs_ntokens(n_tokens)


# let user annotate files
print "please annotate files"

# determine the ntokens setting with the highest accuracy
best_token = None
max_val = 0
accuracy = []
for token in n_tokens:
    data = pd.read_csv(root + "results/" + str(token) + "_head.csv", usecols=["label"])

    current_val = data.label.sum()
    accuracy.append(current_val)
    if current_val > max_val:
        max_val = current_val
        best_token = token
print "Best token is: %i" % best_token

print accuracy

# determine the threshold for the ntokens setting with the highest accuracy
# import al
    # validate tweets above threshold by random sampling

# use the annotations to compare to kmeans
    # run kmeans on cluster


# use the annotations to identify the distribution of the cluster

# use the annotations to show that neural networks per ntokens is better than without ntokens seperation.
# load different ntokens annotations
all_data = pd.DataFrame()
for token in n_tokens:
    head_data = pd.read_csv(root + "results/" + str(token) + "_head.csv", usecols=["id", "label"])
    # check for labels 0 and 1, respectively correct and incorrect if the all tokens neural network performs better
    head_data = head_data[head_data.label == 1]
    token_data = pd.read_csv(root + "results/" + str(token) + ".csv", usecols=["id", "0", "text"])

    data = pd.merge(head_data, token_data, on="id")
    all_data = all_data.append(data)
print all_data["0"].describe()

vectors = pd.read_hdf(vector_path)
vectors=  vectors[vectors.id.isin(all_data.id)][range(70)]

import scipy.stats as stats
import numpy as np
ps = stats.mstats.normaltest(vectors, axis=0).pvalue
print stats.describe(ps)
print len(ps[ps<0.05])

# load all tokens ..
all_tokens_data =  data = pd.read_csv(root + "results/all_tokens.csv", usecols=["id", "0"])
all_tokens_data = all_tokens_data[all_tokens_data.id.isin(all_data.id)]
print all_tokens_data["0"].describe()
# is the probs of all_data different than the tokens_data?
#.. and compare the distribution of the probs for neural network trained with different tokens and all tokens

print stats.ttest_ind(all_data["0"], all_tokens_data["0"], equal_var=False) # False means Welch ttest due to unequal var


# plot loss, accuracy