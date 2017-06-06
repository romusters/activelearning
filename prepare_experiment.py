import config
root, data_path, model_path, vector_path = config.get_paths()
print root
import pandas as pd

def probs_ntokens(n_tokens):
    for token in n_tokens:
        data = pd.read_csv(root + "results/" + str(token) + ".csv", usecols=["id", "text"])
        data["label"] = 0
        data.head(20).to_csv(root + "results/"+ str(token) + "_head.csv")

probs_ntokens(range(1,19))

data = pd.read_csv(root + "results/all_tokens.csv", usecols=["id", "text"])
data["label"] = 0
data.head(20).to_csv(root + "results/all_tokens_head.csv")