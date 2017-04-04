"""
Acquire results from test subjects
- Accuracy on heads and n_tokens
- Inter rater agreement on heads
- Accuracy on 100 head_10 and n_tokens
"""
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.graph_objs import *

import config
root, data_path, model_path, vector_path = config.get_paths()

result_path = "/media/robert/DataUbuntu/Dropbox/Dropbox/Master/proefpersonen/"
import os
directories=[d for d in os.listdir(result_path) if os.path.isdir(result_path + d)]

import pandas as pd
import numpy as np
# Accuracy on 100 head_10 and n_tokens
def acc_100():
    print "All tokens correct count"
    all_accs=[]
    for dir in directories:
        data = pd.read_csv(result_path + dir + "/thresholds/all_tokens_annotate.csv")
        all_accs.append(data["label"].sum())
    print "Head 10 correct count"
    tens_accs = []
    for dir in directories:
        data = pd.read_csv(result_path + dir + "/thresholds/10_annotate.csv")
        tens_accs.append(data["label"].sum())
    import scipy.stats as stats
    print all_accs
    print tens_accs
    print np.mean(all_accs), np.mean(tens_accs)
    print stats.ttest_ind(all_accs, tens_accs, equal_var=False)

# Accuracy on heads
def acc_heads():
    dict = {}
    for dir in directories:
        dict[dir] = None
    print dict
    import numpy as np

    for i in range(1, 19):
        arr = []
        for dir in directories:
            data = pd.read_csv(result_path + dir + "/" + str(i) + "_head.csv")
            # determine how many of the top tweets are about voetbal
            arr.append(data["label"].sum())
        dict[i] = np.array(arr)
    # from scipy import stats
    # means = stats.describe(np.array(dict.values())).mean
    # vars = stats.describe(np.array(dict.values())).variance
    # print means
    # print vars
    print dict
    traces = []
    for i in range(1,19):
        traces.append(Box(y=dict[i], name=str(i) + " tokens" ))

    #all_tokens_head for all users
    arr = []
    for dir in directories:
        data = pd.read_csv(result_path + dir + "/all_tokens_head.csv")
        # determine how many of the top tweets are about voetbal
        arr.append(data["label"].sum())
    dict["all tokens"] = np.array(arr)
    traces.append(Box(y=dict["all tokens"], name="all tokens"))
    data = traces
    layout = Layout(
        title="Number of correct tweets per amount of tokens",
        showlegend=False,
        yaxis=YAxis(
            title="Number of correct tweets"
        ),

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

# compare the average probability for tweets about soccer which are labeled as soccer for each ntoken dataset and all tokendataset
def heads_compared_to_alltokens_using_ntokens():
    dict = {}
    for i in range(1, 19):
        dict[i] = None

    all_tokens_data = pd.read_csv(root + "results/all_tokens.csv")
    all_tokens_data.rename(columns={"0": "all_tokens_probs"}, inplace=True)

    for i in range(1, 19):
        mean_df = pd.DataFrame()
        for dir in directories:
            labeled_ntoken_data = pd.read_csv(result_path + dir + "/" + str(i) + "_head.csv")
            ntoken_data = pd.read_csv(root + "results/" + str(i) + ".csv")
            correct_ids = labeled_ntoken_data[labeled_ntoken_data.label == 1].id
            ntoken_data = ntoken_data[ntoken_data.id.isin(correct_ids)]
            merged = pd.merge(ntoken_data, all_tokens_data, on="id")
            tmp_data = pd.DataFrame({"0": [merged["0"].mean()], "all_tokens_probs": [merged.all_tokens_probs.mean()]})
            mean_df = mean_df.append(tmp_data)
        dict[i] = mean_df
    print dict

    import scipy.stats as stats

    for i,e in dict.iteritems():
        ntoken_data = e["0"]
        all_tokens_data = e["all_tokens_probs"]

        print i, stats.ttest_ind(ntoken_data, all_tokens_data, equal_var=False), all_tokens_data.mean() > ntoken_data.mean()

# compare all labels from all_tokens dataset and group them by amount of tokens. Then compare the probabilities of each group
def compare_all_tokens_vs_ntokens():
    import scipy.stats as stats
    # get all the labels for the all_tokens case
    dict = {}
    for dir in directories:
        dict[dir] = None
    all_data = pd.DataFrame()
    for dir in directories:
        data = pd.read_csv(result_path + dir + "/thresholds/all_tokens_annotate.csv")
        data = data[data.label == 1]
        all_data = all_data.append(data)
    # all_data.drop_duplicates("id")


   # get the all_tokens file with the ntokens column
    data = pd.read_csv(root + "results/all_tokens.csv")
    data.rename(columns={"0": "all_tokens_probs"}, inplace=True)
    grouped = data[data.id.isin(all_data.id)].groupby("ntokens")
    traces = []
    for name, group in grouped:

        ntoken_data = pd.read_csv(root + "results/" + str(name) + ".csv")
        # ntoken_data = ntoken_data[ntoken_data.id.isin(group.id)]
        ntoken_data = pd.merge(ntoken_data, group, on="id")
        all_tokens_probs = group["all_tokens_probs"]
        ntoken_probs = ntoken_data["0"]
        print name, stats.ttest_ind(ntoken_probs, all_tokens_probs, equal_var=False), ntoken_probs.mean() > all_tokens_probs.mean()
        traces.append(Box(y=all_tokens_probs, name=str(name) + " tokens"))
        traces.append(Box(y=ntoken_probs, name=str(name) + " tokens per nn", boxpoints=False, marker= {"color": "black"}))
    data = traces
    layout = Layout(
        title="",
        showlegend=False,
        yaxis=YAxis(
            title="Probability"
        ),

    )
    fig = Figure(data=data, layout=layout)
    plot(fig)

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

def plot_kmeans():
    import pandas as pd
    data = pd.read_csv("/home/robert/lambert/results/kmeans.csv")
    x = data.n_clusters
    y = data.wssse
    print y.values.tolist()
    trace = Scatter(x=x, y=y, mode="markers", marker=dict(color="rgb(0,0,0)"))

    data = [trace]
    layout = Layout(title="WSSSE for different amount of clusters", xaxis=dict(title="Clusters"),
                    yaxis=dict(title="WSSSE"))
    fig = Figure(data=data, layout=layout)
    plot(fig)


def plot_scatter_binary_search_correct_incorrect():
    subject = directories[0]
    path = result_path + subject + "/thresholds/all_tokens_annotate.csv"
    print path
    data = pd.read_csv(path)
    data = data.sort_values("0", ascending=False)
    grouped = data["0"].groupby(data["label"])
    print grouped.get_group(0)

    trace_correct = Scatter(x=grouped.get_group(1).index, y=grouped.get_group(1).values.tolist(), mode="markers", marker=dict(color="rgb(0,255,0)"), name="Correct")
    trace_incorrect = Scatter(x=grouped.get_group(0).index, y=grouped.get_group(0).values.tolist(), mode="markers", marker=dict(color="rgb(255,0,0)"), name="Incorrect")

    data = [trace_correct, trace_incorrect]
    layout = Layout(title="Tweets about voetbal which are classified correctly or <br> incorrectly using the Binary Search method.", xaxis=dict(title="Tweet id"),
                    yaxis=dict(title="Probabilities from the word2vec model"))
    fig = Figure(data=data, layout=layout)
    plot(fig)



def plot_histogram_binary_search_correct_incorrect():
    subject = directories[0]
    path = result_path + subject + "/thresholds/all_tokens_annotate.csv"
    print path
    data = pd.read_csv(path)
    data = data.sort_values("0", ascending=False)
    grouped = data["0"].groupby(data["label"])
    print grouped.get_group(0)

    trace_correct = Histogram(y=grouped.get_group(1).index, x=grouped.get_group(1).values.tolist(), xbins=dict(start=0.9995, end=1.000, size=0.0001),
                            marker=dict(color="rgb(0,255,0)"), opacity=0.75, name="Correct")
    trace_incorrect = Histogram(y=grouped.get_group(0).index, x=grouped.get_group(0).values.tolist(),xbins=dict(start=0.9995, end=1.000, size=0.0001),
                              marker=dict(color="rgb(255,0,0)"), opacity=0.75, name="Incorrect")

    data = [trace_correct, trace_incorrect]
    layout = Layout(
        title="Tweets about voetbal which are classified correctly or <br> incorrectly using the Binary Search method",
        xaxis=dict(title="Probabilities from the word2vec model"),
        yaxis=dict(title="Number of occurrences"),
        barmode="overlay")
    fig = Figure(data=data, layout=layout)
    plot(fig)

def plot_f1():
    data = pd.read_hdf("/home/robert/lambert/metrics.h5")
    print data
    trace = Scatter(x=range(len(data["f1"])), y=data["f1"], name="F1 score",  line = dict(width = 1), opacity= 0.3)
    import numpy as np
    y = np.convolve(data["f1"], np.ones((50,)) / 50, mode="valid")
    trace_smooth = Scatter(x=range(len(data["f1"])), y=y, name="F1 score smoothed", showlegend=False, line=dict(width=1))
    layout = Layout(title="F1 score for voetbal and jihad datasets",
                    xaxis=dict(title="Iterations"), yaxis=dict(title="F1 score"))
    traces = [trace, trace_smooth]
    fig = Figure(data=traces, layout=layout)
    plot(fig)


def plot_loss():
    data = pd.read_hdf("/home/robert/lambert/metrics.h5")
    print data
    trace = Scatter(x=range(len(data["loss"])), y=data["loss"], name="Loss",  line = dict(width = 1), opacity= 0.3)
    import numpy as np
    y = np.convolve(data["loss"], np.ones((10,)) / 10, mode="valid")
    trace_smooth = Scatter(x=range(len(data["loss"])), y=y, name="Loss smoothed", line=dict(width=1))
    layout = Layout(title="Loss for voetbal and jihad datasets",
                    xaxis=dict(title="Iterations"), yaxis=dict(title="Loss"))
    traces = [trace, trace_smooth]
    fig = Figure(data=traces, layout=layout)
    plot(fig)

def plot_train_test():
    data = pd.read_hdf("/home/robert/lambert/metrics.h5")
    import numpy as np
    trace_train = Scatter(x=range(len(data["trainacc"])), y=data["trainacc"], name="Training accuracy", line=dict(width=1), opacity=0.3)
    y = np.convolve(data["trainacc"], np.ones((10,)) / 10, mode="valid")
    trace_train_smooth = Scatter(x=range(len(data["trainacc"])), y=y, name="Training accuracy smoothed", line=dict(width=1))
    trace_test = Scatter(x=range(len(data["testacc"])), y=data["testacc"], name="Testing accuracy", line=dict(width=1),
                    opacity=0.3)
    y = np.convolve(data["testacc"], np.ones((10,)) / 10, mode="valid")
    trace_test_smooth = Scatter(x=range(len(data["testacc"])), y=y, name="Testing accuracy smoothed", line=dict(width=1))
    layout = Layout(title="Train- and test accuracy for voetbal and jihad datasets",
                    xaxis=dict(title="Iterations"), yaxis=dict(title="Accuracy"))

    traces = [trace_train, trace_train_smooth, trace_test, trace_test_smooth]
    fig = Figure(data=traces, layout=layout)
    plot(fig)




import time
# acc_100()
# time.sleep(2)
# acc_heads()
# time.sleep(2)
# acc_all_ntokens()
# time.sleep(2)
# inter_rater()
# time.sleep(2)
# plot_kmeans()
# time.sleep(2)
# import clusters
# clusters.voetbal_in_clusters()
# time.sleep(2)
# plot_histogram_binary_search_correct_incorrect()
# time.sleep(2)
# plot_scatter_binary_search_correct_incorrect()
# time.sleep(2)
# plot_train_test()
# time.sleep(2)
# plot_loss()
# time.sleep(2)
# plot_f1()
# heads_compared_to_alltokens_using_ntokens()
compare_all_tokens_vs_ntokens()