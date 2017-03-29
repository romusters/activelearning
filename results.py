"""
Acquire results from test subjects
- Accuracy on heads and n_tokens
- Inter rater agreement on heads
- Accuracy on 100 head_10 and n_tokens
"""
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.graph_objs import *

result_path = "/media/robert/DataUbuntu/Dropbox/Dropbox/Master/proefpersonen/"
import os
directories=[d for d in os.listdir(result_path) if os.path.isdir(result_path + d)]

import pandas as pd

# Accuracy on 100 head_10 and n_tokens
def acc_100():
    print "All tokens correct count"
    for dir in directories:
        data = pd.read_csv(result_path + dir + "/thresholds/all_tokens_annotate.csv")
        print data["label"].sum()
    print "Head 10 correct count"
    for dir in directories:
        data = pd.read_csv(result_path + dir + "/thresholds/10_annotate.csv")
        print data["label"].sum()


# Accuracy on heads
def acc_heads():
    dict = {}
    for dir in directories:
        dict[dir] = None
    print dict
    import numpy as np

    for dir in directories:
        arr = []
        for i in range(1,19):
            data = pd.read_csv(result_path + dir + "/" + str(i) + "_head.csv")
            arr.append(data["label"].sum())
        dict[dir] = np.array(arr)
    # from scipy import stats
    # means = stats.describe(np.array(dict.values())).mean
    # vars = stats.describe(np.array(dict.values())).variance
    # print means
    # print vars
    traces = []
    for (i, val) in enumerate(dict.values()):
        traces.append(Box(y=val))
    data = traces
    layout = Layout(
        title="Cohen-Kappa score per subject pairs",
    )
    fig = Figure(data=data, layout=layout)
    plot(fig)

# Accuracy on n_tokens
def acc_all_ntokens():
    dict = {}
    for dir in directories:
        dict[dir] = None
    print dict
    import numpy as np

    arr = []
    for dir in directories:
        data = pd.read_csv(result_path + dir + "/all_tokens_head.csv")
        arr.append(data["label"].sum())
        dict[dir] = np.array(arr)
    from scipy import stats
    print stats.describe(np.array(arr))


# Inter rater agreement on heads
def inter_rater():
    dict = {}
    for dir in directories:
        dict[dir] = None
    import numpy as np

    for dir in directories:
        arr = []
        for i in range(1, 19):
            data = pd.read_csv(result_path + dir + "/" + str(i) + "_head.csv")
            arr.append(data["label"].values.tolist())
        dict[dir] = np.array(arr)

    from sklearn.metrics import cohen_kappa_score
    kappas = []
    for (userA, dataA) in dict.iteritems():
        for (userB, dataB) in dict.iteritems():
            dataA = dataA.ravel().ravel()
            dataB = dataB.ravel().ravel()

            kappas.append(cohen_kappa_score(dataA, dataB))

        kappas.append(0)


    n_users = len(dict.items())

    data =  np.array(kappas).reshape((n_users, n_users+1))
    annotations = []
    keys = dict.keys()

    anons = [x[0:2] for x in keys]

    ys = keys
    # ys.append("empty")
    xs = keys
    xs.append("reference")
    print xs
    print ys
    for (x,row) in enumerate(data):
        for (y, v) in enumerate(row):
            v= float("{0:.2f}".format(v))
            annotations.append(Annotation(text=str(v), x=xs[y],y=keys[x], showarrow=False))

    init_notebook_mode()
    trace = Heatmap(z = data, x=xs, y=keys, colorscale=[[0, "red"],[1,"green"]])#"Greys"
    data = [trace]
    layout = Layout(
        title="Cohen-Kappa score per subject pairs",
        xaxis=XAxis(
            title="Anonimized subject"
        ),
        yaxis = YAxis(
            title= "Anonimized subject"
        ),
        showlegend=False,
        height=600,
        width=600,
        annotations=annotations,
    )
    fig = Figure(data=data, layout=layout)
    plot(fig)



# acc_100()
acc_heads()
# acc_all_ntokens()
# inter_rater()