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
    # print freqs
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot
    from plotly.graph_objs import *

    trace = Scatter(x=freqs.keys(), y=freqs.values(), mode="markers", marker=dict(color="rgb(0,0,0)"))

    data_plotly = [trace]
    layout = Layout(title="Amount of datapoints in each Kmeans cluster", xaxis=dict(title="Cluster number"),
                    yaxis=dict(title="Number of datapoints in cluster"))
    fig = Figure(data=data_plotly, layout=layout)
    plot(fig)


def ratio_soccer_notsoccer():
    import numpy as np
    #take a clustering according to a certain k
    data = pd.read_csv(root + "clusters.csv", header=None)
    data["id"] = range(len(data.index))

    # get the labels
    import experiment
    soccer = experiment.get_all_voetbal()
    not_soccer =  experiment.get_all_not_voetbal()

    from collections import Counter
    # for each cluster, determine the ratio of soccer and not soccer in the cluster
    freqs_soccer = data[data.id.isin(soccer)][0].values.tolist()
    freqs_soccer = Counter(freqs_soccer)
    freqs_soccer = sorted(freqs_soccer.items(), key=lambda pair: pair[0], reverse=False)
    print freqs_soccer

    freqs_not_soccer = data[data.id.isin(not_soccer)][0].values.tolist()
    freqs_not_soccer = Counter(freqs_not_soccer)
    freqs_not_soccer = sorted(freqs_not_soccer.items(), key=lambda pair: pair[0], reverse=False)
    print freqs_not_soccer

    both = zip(freqs_soccer, freqs_not_soccer)
    print both
    result = []
    for e in both:
        if e[1][1] == 0:
            result.append(np.inf)
        else:
            result.append(e[0][1]/e[1][1])
    print max(result)
    print result.index(max(result))
# voetbal_in_clusters()

ratio_soccer_notsoccer()