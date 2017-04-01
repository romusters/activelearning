"""
The kmeans clusters are downloaded. Now we need to determine how many members there are in each cluster.
We also need to check which voetbal tweets lie in which clusters.
If the voetbal tweets all lie in one cluster, this is better.

"""


import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import config
root, data_path, model_path, vector_path = config.get_paths()
print root

import pandas as pd
data = pd.read_csv(root + "clusters.csv", header = None)

def plot_kmeans_clusters_members():
    clusters = data.values.tolist()
    clusters = [x[0] for x in clusters]

    from collections import Counter
    freqs = Counter(clusters)
    print freqs.values()
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot
    from plotly.graph_objs import *

    trace = Scatter(x=freqs.keys(), y=freqs.values(), mode="markers", marker=dict(color="rgb(0,0,0)"))

    data = [trace]
    layout = Layout(title="Amount of datapoints in each Kmeans cluster", xaxis=dict(title="Cluster number"),
                    yaxis=dict(title="Number of datapoints in cluster"))
    fig = Figure(data=data, layout=layout)
    plot(fig)


def voetbal_in_clusters():
    data["id"] = range(len(data.index))
    import experiment
    voetbal_ids = experiment.get_all_voetbal()
    clusters = data[data.id.isin(voetbal_ids)][0].values.tolist()
    print "Amount of clusters with voetbal members: %i" % len(set(clusters))

    from collections import Counter
    freqs = Counter(clusters)
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot
    from plotly.graph_objs import *

    trace = Scatter(x=freqs.keys(), y=freqs.values(), mode="markers", marker=dict(color="rgb(0,0,0)"))

    data_plotly = [trace]
    layout = Layout(title="Amount of datapoints in each Kmeans cluster", xaxis=dict(title="Cluster number"),
                    yaxis=dict(title="Number of datapoints in cluster"))
    fig = Figure(data=data_plotly, layout=layout)
    plot(fig)
# voetbal_in_clusters()