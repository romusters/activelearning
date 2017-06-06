import pandas as pd
def get_sample():
    data = pd.read_csv("/home/robert/lambert/datasets/data.csv", names = ["text", "filtered_text", "id"])
    print len(data.index)
    # remove duplicate textual entries
    data.drop_duplicates("filtered_text",  inplace=True)
    print len(data.index)
    data = data.drop("id", axis=1)
    data = data.sample(frac=0.2, replace=False)
    data["id"] = range(len(data.index))
    data.to_csv("/home/robert/lambert/datasets/data_sample.csv")

def inspect_sample():
    data = pd.read_csv("/home/robert/lambert/datasets/data_sample.csv", usecols=["id", "text", "filtered_text"])
    print data
    print len(data.index)
    data.drop_duplicates("id", inplace=True)
    # remove duplicate textual entries
    print len(data.index)
    print data.index.get_duplicates()
    data.to_csv("/home/robert/lambert/datasets/data_sample.csv")


import config
root, data_path, model_path, vector_path = config.get_paths()
get_sample()
inspect_sample()
import w2v
w2v.get_vectors(data_path + "data_sample.csv", vector_path, model_path + "w2vmodel")
#
# data = pd.read_csv("/home/robert/lambert/datasets/data_sample.csv", usecols=["text", "filtered_text", "id"])
# print type(data.id[0])