import os
import sys
import logging
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class Dataset:

    def __init__(self, root):
        import config
        root, data_path, model_path, vector_path = config.get_paths()
        self.root = root
        self.data_path = data_path
        self.model_path = model_path
        self.vector_path = vector_path
        self.all_hashtags = ["voetbal", "moslim", "werk", "economie","jihad", "seks", "politiek"]  #
        self.all_vectors_store = pd.HDFStore(self.root + "w2v_vector.h5")
        self.balanced_store = pd.HDFStore(self.root + "datasets/seeds/balanced.h5")
        self.tweets = pd.read_csv(self.root + "datasets/data_sample.csv")

    def set_root(self, root):
        self.root = root

    def get_root(self):
        return self.root

    # def transform_dataset(self, data, n_classes):
    # 	logger.info("Transforming data for %i classes", n_classes)
    # 	import numpy as np
    # 	import features
    # 	data = data.dropna()
    # 	trainset = data.sample(frac=0.8, random_state=200)
    # 	testset = data.drop(trainset.index)
    #
    # 	result = {}
    # 	result["trainset"] = np.array(trainset[range(70)].values.tolist())
    # 	result["trainlabels"] = np.array(trainset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
    # 	result["trainids"] = np.array(trainset.id.values.tolist())
    # 	result["testdata"] = np.array(testset[range(70)].values.tolist())
    # 	result["testlabels"] = np.array(testset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
    # 	result["testids"] = np.array(testset.id.values.tolist())
    # 	result["nclasses"] = n_classes
    # 	result["allvectors"] = np.array(self.all_vectors_store["data"][range(70)].values.tolist())
    # 	result["allvectorsids"] = np.array(self.all_vectors_store["data"]["id"].values.tolist())
    #
    # 	return result

    def transform_dataset(self, data, i, n_classes):
        logger.info("Transforming data for %i classes", n_classes)
        import numpy as np
        import features
        data = data.dropna()

        data["labels"] = data["labels"].replace(i, 0)
        data["labels"] = data["labels"].replace(i+1, 1)
        data = data.sample(frac=1)
        logger.info("sampling")
        data.index = range(0, len(data.index)) # needed because some sets are so large, all the ids get removed in the testset
        logger.info("Generating train and testset.")
        trainset = data.sample(frac=0.8, random_state=200)
        testset = data.drop(trainset.index)

        logger.info("Generating onehot enoding.")
        result = {}
        result["trainset"] = np.array(trainset[range(70)].values.tolist())
        result["trainlabels"] = np.array(trainset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
        result["trainids"] = np.array(trainset.id.values.tolist())
        result["testdata"] = np.array(testset[range(70)].values.tolist())
        result["testlabels"] = np.array(testset["labels"].apply(lambda x: features.onehot(x, n_classes - 1)).values.tolist())
        result["testids"] = np.array(testset.id.values.tolist())
        result["nclasses"] = n_classes
        tmp_data = self.all_vectors_store["data"].sample(n=43000)
        result["allvectors"] = np.array(tmp_data[range(70)].values.tolist())
        result["allvectorsids"] = np.array(tmp_data["id"].values.tolist())

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

    def make_probs_file(self, hashtag, seed_id, column_id=None):
        import pandas as pd
        data = pd.read_hdf(self.root + "results/seeds/" + str(seed_id) + "/probs.h5")
        probs = data[column_id]
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

    def create_subject_sets(self):
        logging.info("Tweet path : %s", self.data_path)
        print "%i tweets read" % self.tweets.count()[0]

        for idx, hashtag in enumerate(self.all_hashtags):
            print hashtag

            hashtag_set = self.tweets[self.tweets.filtered_text.str.contains( hashtag)] # Or add # to only select hashtags
            hashtag_set.index = hashtag_set["id"]
            vectors = self.all_vectors_store.select("data", where=self.all_vectors_store["data"]["id"].isin(hashtag_set.id))
            print "%i tweets found for subject %s" % (len(vectors.index), hashtag)
            # create label
            vectors["labels"] = idx
            vectors.index = vectors["id"]
            hashtag_set.to_csv(self.root + "datasets/seeds/" + hashtag + ".csv")
            vectors.to_hdf(self.root + "datasets/seeds/" + hashtag + ".h5", "data", format="table")
            print "Subject %s written." % hashtag

    def load_all_data(self):
        all = pd.DataFrame()
        for hashtag in self.all_hashtags:
            tmp = pd.HDFStore(self.root + "datasets/seeds/" + hashtag + ".h5")["data"]
            tmp.index = tmp.id
            print len(tmp.id)
            all = all.append(tmp)
        all_tweets = pd.read_csv(self.data_path + "data_sample.csv")
        # print all_tweets
        all_ids = all_tweets["id"]
        print len(all.index)
        print len(all_ids)
        return all.sample(frac=1)

    def balance_data(self):
        logging.info("Tweet path : %s", self.data_path)
        data = self.load_all_data()
        # drop data with more than one instance of an id, because these are overlapping subjects
        data = data.drop_duplicates("id", keep=False)
        size_largest_label = data.groupby("labels").count().id.max()  # 42869
        print "largest label is: %i" % size_largest_label

        balanced_data = pd.DataFrame()
        for label in set(data["labels"]):
            category_data = data[data["labels"] == label]
            new_data = category_data.sample(n=size_largest_label, replace=True)
            print "size of balanced subject data %i", new_data.id.count()
            balanced_data = balanced_data.append(new_data)
        balanced_data = balanced_data.sample(frac=1)
        print len(balanced_data.index)
        balanced_data.to_hdf(self.root + "datasets/seeds/balanced.h5", "data")

def create_sample(data_path):
    data = pd.read_csv(data_path + "data.csv", names = ["text", "filtered_text", "id"])
    df = data.dropna()
    print "Data size before deduplification: %i" % len(df.index)
    df = df.drop_duplicates("id")
    print "Data size after deduplification: %i" % len(df.index)

    print "Data size before nan drop: %i " % len(df.index)
    df = df.dropna()
    print "Data size after nan drop: %i" % len(df.index)
    # data.index = data.id
    # data = data.drop("id", axis=1)
    # print data
    sample = df.sample(frac=0.2)
    sample.to_csv(data_path + "data_sample.csv")


# dset = Dataset("/home/robert/lambert/")
# dset.create_subject_sets()


