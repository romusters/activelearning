import config
root, data_path, model_path, vector_path = config.get_paths()

import pandas as pd
data = pd.read_csv(data_path + "data.csv", names=["text", "filtered", "id"])

# number of line is
# print data.count()
# 8175231

# original
# results = set()
# data["text"].str.split().apply(results.update)
# print len(results)
# 5430934

# tolower
# results = set()
# data["text"].str.lower().str.split().apply(results.update)
# print len(results)
# 5020577


# token mention
# import string
# results = set()
# data["text"] = data['text']\
#     .apply(lambda x:''.join([i for i in x if i not in string.punctuation]))\
#     .str.lower().str.split()\
#     .apply(lambda x: [e for e in x if "@" not in e])\
#
# data["text"].apply(results.update)
# print len(results)
# 3769888

# remove punctuation
import string
results = set()
data["text"] = data['text']\
    .apply(lambda x: [e for e in x if "@" not in e]) \
    .apply(lambda x:''.join([i for i in x if i not in string.punctuation]))\
    .str.lower().str.split().apply(results.update)
print len(results)
3769888?!


# stem
# import string
# results = set()
# data["text"] = data['text']\
#     .apply(lambda x:''.join([i for i in x if i not in string.punctuation]))\
#     .str.lower().str.split()\
#     .apply(lambda x: [e for e in x if "@" not in e]).apply(results.update)
# print len(results)



# token url

# token