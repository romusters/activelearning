class Dataset:
	import pandas as pd
	all_hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]

	all_vectors_store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	balanced_store = pd.HDFStore("/media/cluster/data1/lambert/datasets/seeds/balanced.h5")


	def get_dataset(self, labels):
		import numpy as np
		import features
		data = self.balanced_store["data"]
		data = data[data["labels"].isin(labels)]
		data = data.dropna()
		trainset = data.sample(frac=0.8, random_state=200)
		testset = data.drop(trainset.index)

		result = {}
		n_classes = len(labels)
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


	def save_meta_dataset(self, path, hashtag, i, threshold):
		import pandas as pd
		store = pd.read_hdf("/media/cluster/data1/lambert/results/" + hashtag + ".h5")
		data = store.select("data", where=store["data"]["probs"] > threshold)
		meta_dataset = self.all_vectors_store.select("data", where=self.all_vectors_store["data"]["id"].isin(data.id))
		meta_dataset["labels"] = i
		# Save metadataset
		meta_dataset.to_hdf("data", "/media/cluster/data1/lambert/dataset/meta_subject/" + hashtag + ".h5")


	def get_meta_dataset(path, hashtag):
		import pandas as pd
		meta_dataset = pd.read_hdf("data", "/media/cluster/data1/lambert/dataset/meta_subject/" + hashtag + ".h5")
		return meta_dataset


	def get_seed_dataset(self, labels):
		'''
		Get the seed dataset returning vectors and ids

		:return Dataframe with vectors and ids.

		'''
		import numpy as np
		data = self.balanced_store.select("data", where=self.balanced_store["data"].labels.isin(labels))
		return data


	def get_meta_seed_dataset(self, path, metas, seed):
		'''
		Get the seed dataset and combine it with the metadataset

		:param path:
		:param metas:
		:param seed:
		:return: Dataframe containing vectors and ids.
		'''
		import pandas as pd
		dataset = pd.DataFrame()
		seed_dataset = self.get_seed_dataset(seed)
		meta_dataset = self.get_meta_dataset(metas)
		dataset = meta_dataset.append(seed_dataset)
		return dataset