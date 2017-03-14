import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def w2vmodel_to_hdf(model_path):
    logger.info("Converting w2vmodel.csv to hdf5.")
    data = pd.read_csv(model_path + ".csv", names=["words", "vectors"])
    data["vectors"] = data["vectors"].apply(lambda x: eval(x.replace("WrappedArray(", "[").replace(")", "]")))
    data.to_hdf(model_path.replace(".csv", ".h5"), "data")


def get_vectors(data_name, vector_name, model_name):
    logger.info("Calculating w2v vectors for data.")
    model = pd.read_hdf(model_name + ".h5")
    dictionary = dict(zip(model.words, model.vectors))

    store = pd.HDFStore(vector_name)
    chunksize = 10000
    data = pd.read_csv(data_name, header=None, names=["text", "filtered_text", "id"], iterator=True,
                       chunksize=chunksize)
    chunk = data.get_chunk()

    idx = 0
    while chunk is not None:
        print idx
        chunk = chunk.dropna()
        vectors = []

        all_tokens = chunk.filtered_text.apply(lambda x: x.split())
        for tokens in all_tokens:
            tmp_vector = []
            for token in tokens:
                if token not in ["<stopword>", "<url>", "rt", "<mention>"]:
                    try:
                        tmp_vector.append(dictionary[token])
                    except KeyError:
                        pass
            vectors.append(np.mean(tmp_vector, axis=0))
        df = pd.DataFrame({"vectors": vectors})
        df = df.vectors.apply(pd.Series, 1)
        df["id"] = chunk["id"].apply(lambda x: int(x)).values.tolist()
        df = df.dropna()
        store.append("data", df)
        chunk = data.get_chunk()
        idx += 1
    store.close()

def test_model(fname):
    import features
    data = pd.HDFStore(fname + ".h5")["data"]
    words = ["voetbal", "moslim", "1", "porno", "advertentie"]
    for word in words:
        print word.upper()
        words = data["words"].values.tolist()
        print words[0]
        print word in words
        A = data[data["words"] == word].vectors.values.tolist()[0]
        print A
        sims = data.vectors.apply(lambda x: features.cosine_similarity(np.array(x), np.array(A)))
        sims.to_csv(fname + ".sims")

        sims.sort(ascending=False)
        els = sims.head(20).keys()

        print els

        for el in els:
            print data.words[el]