import os
import sys
import logging
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class Dataset:

	def __init__(self, root):
		self.root = root
		self.all_hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]
		self.all_vectors_store = pd.HDFStore(self.root + "data_sample_vector_id.clean.h5")
		self.balanced_store = pd.HDFStore(self.root + "datasets/seeds/balanced.h5")
		self.tweets = pd.read_csv(self.root + "lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])

	def set_root(self, root):
		self.root = root

	def get_root(self):
		return self.root

	def transform_dataset(self, data, n_classes):
		logger.info("Transforming data for %i classes", n_classes)
		import numpy as np
		import features
		data = data.dropna()
		trainset = data.sample(frac=0.8, random_state=200)
		testset = data.drop(trainset.index)

		result = {}
		result["trainset"] = np.array(trainset[range(70)].values.tolist())
		result["trainlabels"] = np.array(trainset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
		result["trainids"] = np.array(trainset.id.values.tolist())
		result["testdata"] = np.array(testset[range(70)].values.tolist())
		result["testlabels"] = np.array(testset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
		result["testids"] = np.array(testset.id.values.tolist())
		result["nclasses"] = n_classes
		result["allvectors"] = np.array(self.all_vectors_store["data"][range(70)].values.tolist())
		result["allvectorsids"] = np.array(self.all_vectors_store["data"]["id"].values.tolist())

		return result


	def transform_dataset_meta(self, data, n_classes):
		import numpy as np
		import features
		data = data.dropna()
		data.index = data["id"]
		print "size of pos data is: %i" % len(data.index)
		self.all_vectors_store.index = self.all_vectors_store["data"]["id"]
		all_data = self.all_vectors_store["data"].dropna()

		# Get the data not containing ids from the meta subjects
		neg_data = all_data.drop(data.index).sample(n=len(data.index))
		# maybe if there is a big overlap between the subjects, merge them?
		print "size of neg data is: %i" % len(neg_data.index)

		neg_data["labels"] = n_classes -1
		dataset = data.append(neg_data).sample(frac=1)
		trainset = dataset.sample(frac=0.8, random_state=200)
		testset = dataset.drop(trainset.index)

		result = {}
		result["trainset"] = np.array(trainset[range(70)].values.tolist())
		# if a label group X contains tweets from a previous label group B, group X should be one hot encoded with multiple labels
		result["trainlabels"] = np.array(trainset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
		result["trainids"] = np.array(trainset.id.values.tolist())
		result["testdata"] = np.array(testset[range(70)].values.tolist())
		result["testlabels"] = np.array(testset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
		result["testids"] = np.array(testset.id.values.tolist())
		result["nclasses"] = n_classes
		result["allvectors"] = np.array(self.all_vectors_store["data"][range(70)].values.tolist())
		result["allvectorsids"] = np.array(self.all_vectors_store["data"]["id"].values.tolist())

		return result



	def save_meta_dataset(self, hashtag, threshold, hashtag_label):
		import pandas as pd
		data = pd.read_csv(self.root + "results/probs/" + hashtag + ".csv", usecols=["id", "probs", "text"])
		data = data.dropna()
		data["probs"] = data["probs"].apply(lambda x: float(x))
		ids = data[data["probs"] > threshold].id
		vectors = self.all_vectors_store.select("data", where=self.all_vectors_store["data"]["id"].isin(ids))
		vectors["labels"] = hashtag_label
		vectors.to_hdf(self.root + "datasets/meta_subjects/" + hashtag + ".h5", "data")


	def get_meta_dataset(self, hashtags):
		import pandas as pd
		meta_dataset = pd.read_hdf(self.root + "datasets/meta_subjects/" + hashtags[0] + ".h5", "data")

		# Balance the data so that the largest group is taken as the reference group
		reference_size = len(meta_dataset.index)
		dataset = meta_dataset
		for hashtag in hashtags[1:]:
			meta_dataset = pd.read_hdf( self.root + "datasets/meta_subjects/" + hashtag + ".h5", "data")
			meta_dataset = meta_dataset.sample(n=reference_size, replace=True)
			dataset = dataset.append(meta_dataset)

		return dataset

	def make_probs_file(self, hashtag, i):
		import pandas as pd
		data = pd.read_hdf(self.root + "results/seeds/" + str(i) + "/probs.h5")
		probs = data[i]
		ids = data.id
		df = pd.DataFrame({"id": ids, "probs": probs, "text": self.tweets.text})
		df = df.sort_values(by=["probs"], ascending=False) #TEST!
		df.to_csv(self.root + "results/probs/" + hashtag + ".csv")


	def get_seed_dataset(self, labels):
		'''
		Get the seed dataset returning vectors and ids

		:return Dataframe with vectors and ids.

		'''
		import numpy as np
		data = self.balanced_store["data"]
		data = data[data["labels"].isin(labels)]
				# .select("data", where=self.balanced_store["data"]["labels"].isin(labels))
		return data


	def get_meta_seed_dataset(self, metas, seed):
		'''
		Get the seed dataset and combine it with the metadataset

		:param path:
		:param metas:
		:param seed:
		:return: Dataframe containing vectors and ids.
		'''
		meta_dataset = self.get_meta_dataset(metas)
		reference_size = len(meta_dataset.index)
		seed_dataset = self.get_seed_dataset(seed)
		seed_dataset = seed_dataset.sample(n=reference_size, replace=True)
		dataset = meta_dataset.append(seed_dataset)
		return dataset

	def make_seed_datasets(self):
		# dataset = self.tweets[self.tweets.text.str.contains(r'\b(%s|%s)\b' % ("baby", "peuter"))]
		dataset = self.tweets[self.tweets.text.str.contains(r'\b(%s|%s)\b' % ("bagage", "tas"))]
		# dataset = self.tweets[self.tweets.text.str.lower().str.contains(r'\b(%s)\b' % ("bouwmaterialen"))]		dataset = self.tweets[self.tweets.text.str.lower().str.contains(r'\b(%s)\b' % ("bouwmaterialen"))]


		print dataset.text.values