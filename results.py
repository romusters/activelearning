"""
Acquire results from test subjects
- Accuracy on heads and n_tokens
- Inter rater agreement on heads
- Accuracy on 100 head_10 and n_tokens
"""
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.graph_objs import *

import time

import config
root, data_path, model_path, vector_path = config.get_paths()

result_path = "/media/robert/DataUbuntu/Dropbox/Dropbox/Dropbox/Master/proefpersonen/"
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
        title="Number of correct top 20 tweets with highest softmax value per amount of tokens",
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


    heads_df = pd.DataFrame()
    for i in range(1, 19):
        for dir in directories:
            data = pd.read_csv(result_path + dir + "/" + str(i) + "_head.csv")
            data = data[data.label == 1]
            heads_df = heads_df.append(data)

    all_data = all_data.append(heads_df)
    all_data.drop_duplicates("id")
   # get the all_tokens file with the ntokens column
    data = pd.read_csv(root + "results/all_tokens.csv")
    data.rename(columns={"0": "all_tokens_probs"}, inplace=True)
    # grouped = data[data.id.isin(all_data.id)].groupby("ntokens")
    grouped = pd.merge(all_data, data, on="id").groupby("ntokens")
    traces = []
    ntoken_probs_mean = []
    all_tokens_probs_mean = []
    ttests = []
    biggers = []
    counts = []
    for name, group in grouped:
        ntoken_data = pd.read_csv(root + "results/" + str(name) + ".csv")

        # ntoken_data = ntoken_data[ntoken_data.id.isin(group.id)]
        ntoken_data = pd.merge(ntoken_data, group, on="id")
        all_tokens_probs = group["all_tokens_probs"]
        # print ntoken_data
        ntoken_probs = ntoken_data["0_x"]
        t_test = stats.ttest_ind(ntoken_probs, all_tokens_probs, equal_var=False)
        # print name, all_tokens_probs.count(), stats.ttest_ind(ntoken_probs, all_tokens_probs, equal_var=False), ntoken_probs.mean() > all_tokens_probs.mean()
        ttests.append(float(t_test[1]))
        biggers.append(all_tokens_probs.mean() > ntoken_probs.mean())
        ntoken_probs_mean.append(ntoken_probs.mean())
        all_tokens_probs_mean.append( all_tokens_probs.mean())
        counts.append(all_tokens_probs.count())
        #
        # traces.append(Box(y=all_tokens_probs, name=str(name) + " tokens"))
        # traces.append(Box(y=ntoken_probs, name=str(name) + " tokens per nn", boxpoints=False, marker= {"color": "black"}))

    import plotly.figure_factory as ff
    data_matrix = [["n tokens", "p-value", "All tokens", "N tokens", "All > N", "N samples"]]

    import numpy as np
    # ttests = np.array(ttests)
    ttests = [0.33, 0.18, "0.00", 0.06, 0.16, "0.00", 0.19, "0.00", "0.00", "0.00", "0.00", 0.24, "0.00", 0.21, 0.41, "0.00", 0.77, 0.21]
    data_matrix.extend(zip(range(19), ttests, all_tokens_probs_mean, ntoken_probs_mean, biggers, counts))
    print data_matrix
    table = ff.create_table(data_matrix, height_constant=20)
    plot(table)


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

    print ntoken_probs_mean
    print all_tokens_probs_mean
    traces.append(Scatter(x=range(1,19), y=ntoken_probs_mean, name="N tokens"))
    traces.append(Scatter(x=range(1,19), y=all_tokens_probs_mean, name="All tokens"))
    data = traces
    layout = Layout(
        title="Comparison between performance of neural network trained on all tokens and N tokens",
        showlegend=True,
        xaxis=XAxis(
            title="N tokens"
        ),
        yaxis=YAxis(
            title="Probabilities"
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
    layout = Layout(title="WSSSE for different amount of clusters", xaxis=dict(title="K"),
                    yaxis=dict(title="WSSSE"))
    fig = Figure(data=data, layout=layout)
    plot(fig)


def plot_scatter_binary_search_correct_incorrect():
    all_data = pd.DataFrame()
    for dir in directories:
        path = result_path + dir + "/thresholds/10_annotate.csv"
        print path
        data = pd.read_csv(path)
        all_data = all_data.append(data)
    all_data = all_data.sort_values("0", ascending=False)
    grouped = all_data["0"].groupby(all_data["label"])
    _10_group_0 = grouped.get_group(0).sort_values().values.tolist()
    _10_group_1 = grouped.get_group(1).sort_values().values.tolist()
    print len(_10_group_0), len(_10_group_1)
    # trace_correct = Scatter(x=range(len(group_1.index)), y=group_1, mode="markers", marker=dict(color="rgb(0,255,0)"), name="Correct")
    # trace_incorrect = Scatter(x=range(len(group_0.index)), y=group_0, mode="markers", marker=dict(color="rgb(255,0,0)"), name="Incorrect")
    #
    # data = [trace_correct, trace_incorrect]
    # layout = Layout(title="Tweets about voetbal which are classified correctly or <br> incorrectly using the Binary Search method.", xaxis=dict(title="Tweet id"),
    #                 yaxis=dict(title="Probabilities from the word2vec model"))
    # fig = Figure(data=data, layout=layout)
    all_data = pd.DataFrame()
    for dir in directories:
        path = result_path + dir + "/thresholds/all_tokens_annotate.csv"
        print path
        data = pd.read_csv(path)
        all_data = all_data.append(data)
    all_data = all_data.sort_values("0", ascending=False)
    grouped = all_data["0"].groupby(all_data["label"])
    all_group_0 = grouped.get_group(0).sort_values().values.tolist()
    all_group_1 = grouped.get_group(1).sort_values().values.tolist()
    print len(all_group_0), len(all_group_1)
    # normalize data
    # from sklearn import preprocessing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # _10_group_0 = min_max_scaler.fit_transform(_10_group_0.values.tolist())
    # _10_group_1 = min_max_scaler.fit_transform(_10_group_1.values.tolist())
    # all_group_0 = min_max_scaler.fit_transform(all_group_0.values.tolist())
    # all_group_1 = min_max_scaler.fit_transform(all_group_1.values.tolist())

    # import seaborn as sns
    # from scipy.stats import norm
    # # sns.distplot(_10_group_0, kde=False, hist=False, fit=norm)
    # # sns.distplot(_10_group_1, kde=False, hist=False, fit=norm)
    # sns.distplot(all_group_0, kde=False, hist=False)
    # # sns.distplot(all_group_1, kde=False, hist=False, fit=norm)
    # sns.plt.show()
    # import sys
    # sys.exit(0)

    import plotly.figure_factory as ff
    hist_data = [_10_group_0, _10_group_1]
    colors = ["red", "green", "blue", "orange"]
    group_labels = ['10 tokens incorrect', '10 tokens correct']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors, bin_size=1)
    fig['layout'].update(title='Distribution of softmax values for 10 tokens method', yaxis=dict(title="Frequency"), xaxis=dict(title="Hierarchical softmax value"))
    plot(fig)

    import time
    time.sleep(2)
    import plotly.figure_factory as ff
    hist_data = [all_group_0, all_group_1]
    colors = ["red", "green", "blue", "orange"]
    group_labels = ['All tokens incorrect', 'All tokens correct']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)
    fig['layout'].update(title='Distribution of softmax values for all tokens method', yaxis=dict(title="Frequency"),
                         xaxis=dict(title="Hierarchical softmax value"))
    plot(fig)
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import matplotlib.mlab as mlab
    # import math
    # all_group_0 = all_group_1[:-300]
    # print all_group_0
    # mu = np.array(all_group_0).mean()
    # print mu
    # variance = np.array(all_group_0).var()
    # sigma = math.sqrt(variance)
    # print sigma
    # x  = np.arange(0.999, 1, 0.0001)
    # plt.plot(x, mlab.normpdf(x, mu, sigma))
    #
    # plt.show()


def plot_histogram_binary_search_correct_incorrect():
    all_data = pd.DataFrame()
    for dir in directories:
        path = result_path + dir + "/thresholds/all_tokens_annotate.csv"
        print path
        data = pd.read_csv(path)
        all_data = all_data.append(data)
    print all_data
    all_data = all_data.sort_values("0", ascending=False)
    grouped = all_data["0"].groupby(all_data["label"])
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


def prec_recall_thresholds():
    # show 10_annotate and all_tokens_annotate in a single figure
    all_data = pd.DataFrame()
    for dir in directories:
        path = result_path + dir + "/thresholds/10_annotate.csv"
        data = pd.read_csv(path)
        all_data = all_data.append(data)
    print all_data
    all_data = all_data.sort_values("0", ascending=False)
    grouped = all_data["0"].groupby(all_data["label"])
    group_0 = grouped.get_group(0)
    group_1 = grouped.get_group(1)
    print all_data["0"].min()
    # for different thresholds
    thresholds = range(int((1.000000 - all_data["0"].min())*1000000)+2)
    thresholds = [(0.000001 *x)+all_data["0"].min() for x in thresholds]
    print thresholds
    ps = []
    rs = []
    f1s = []
    accs = []
    for t in thresholds:
        true_positives = group_1[group_1.values > t].count()
        false_positives = group_0[group_0.values > t].count()
        # true negatives are not used, which is a flaw
        true_negatives = group_0[group_0.values < t].count()
        false_negatives = group_1[group_1.values < t].count()

        if float(true_positives + false_positives) == 0:
            precision = true_positives / 0.000001
            continue

        else:
            precision = true_positives / float(true_positives + false_positives)

        if true_positives+false_negatives == 0:
            recall = float(true_positives / 0.00000001)
            continue
        else:
            recall = float(true_positives/float(true_positives+false_negatives))


        if precision + recall == 0:
            f1 = 2 * ((precision * recall) / 0.000001)
            continue
        else:
            f1 = 2 * ((precision * recall) / (precision + recall))

        if (float(true_negatives) + false_negatives) == 0:
            acc = ((true_positives + false_positives)/(0.0000001))/len(all_data.index)
            continue
        else:
            acc = ((true_positives + false_positives) / (float(true_negatives) + false_negatives)) / len(all_data.index)

        ps.append(precision)
        rs.append(recall)
        f1s.append(f1)
        accs.append(acc)

    # for all_tokens_annotate
    all_data = pd.DataFrame()
    for dir in directories:
        path = result_path + dir + "/thresholds/all_tokens_annotate.csv"
        print path
        data = pd.read_csv(path)
        all_data = all_data.append(data)
    all_data = all_data.sort_values("0", ascending=False)
    grouped = all_data["0"].groupby(all_data["label"])
    group_0 = grouped.get_group(0)
    group_1 = grouped.get_group(1)

    # for different thresholds
    thresholds = range(int((1.000000 - 0.999526)*1000000)+2)
    thresholds = [(0.000001 *x)+0.999526 for x in thresholds]

    all_tokens_ps = []
    all_tokens_rs = []
    all_tokens_f1s = []
    all_tokens_accs = []
    for t in thresholds:
        true_positives = group_1[group_1.values > t].count()
        false_positives = group_0[group_0.values > t].count()
        # true negatives are not used, which is a flaw
        true_negatives = group_0[group_0.values < t].count()
        false_negatives = group_1[group_1.values < t].count()
        if float(true_positives + false_positives) == 0:
            precision = true_positives / 0.000001
            continue
        else:
            precision = true_positives / float(true_positives + false_positives)
        # precision = true_positives/float(true_positives + false_positives)
        if float(true_positives+false_negatives) == 0.0:
            recall = float(true_positives / 0.0000001)
            print "error"
            continue
        else:
            recall = float(true_positives/float(true_positives+false_negatives))

        f1 = 2* ((precision * recall)/(precision + recall))
        acc = ((true_positives + false_positives)/(float(true_negatives) + false_negatives))/len(all_data.index)
        all_tokens_ps.append(precision)
        all_tokens_rs.append(recall)
        all_tokens_f1s.append(f1)
        all_tokens_accs.append(acc)





    trace = Scatter(x=ps, y=rs, mode="markers", name="10 tokens")
    all_tokens_trace = Scatter(x=all_tokens_ps, y=all_tokens_rs, mode="markers", name="All tokens")
    data = [trace, all_tokens_trace]
    layout = Layout(
        title="Precision and Recall using Binary Search method for two different methods",
        xaxis=dict(title="Recall"),
        yaxis=dict(title="Precision"))
    fig = Figure(data=data, layout=layout)
    plot(fig)
    time.sleep(2)
    #
    # trace = Scatter(x=thresholds, y=f1s, mode="markers", name="10 tokens")
    # all_tokens_trace = Scatter(x=thresholds, y=all_tokens_f1s, mode="markers", name="All tokens")
    #
    # data = [trace, all_tokens_trace]
    # layout = Layout(
    #     title="F1 score using Binary Search method for two different methods.",
    #     xaxis=dict(title="Threshold"),
    #     yaxis=dict(title="F1 score"))
    # fig = Figure(data=data, layout=layout)
    # plot(fig)
    # time.sleep(2)
    # print "ACCS", accs
    # trace = Scatter(x=thresholds, y=accs, mode="markers", name="10 tokens")
    # all_tokens_trace = Scatter(x=thresholds, y=all_tokens_accs, mode="markers", name="All tokens")
    #
    # data = [trace, all_tokens_trace]
    # layout = Layout(
    #     title="Accuracy using Binary Search method for two different methods.",
    #     xaxis=dict(title="Threshold"),
    #     yaxis=dict(title="Accuracy"))
    # fig = Figure(data=data, layout=layout)
    # plot(fig)

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
    layout = Layout(title="Loss for voetbal and jihad data",
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


def plot_cluster_member_ratio_and_max():
    data = [(2, 351), (35, 137), (7, 86), (9, 52), (33, 53), (15, 45), (18, 35), (26, 35), (30, 30), (23, 24), (24, 24), (18, 24), (15, 24), (17, 19), (13, 18), (14, 18), (16, 19), (11, 15), (13, 17), (15, 17), (12, 17), (14, 14), (14, 14), (13, 14), (10, 13), (9,12)]
    ratios = []
    max = []
    for e in data:
        ratios.append(e[0])
        max.append(e[1])
    _range = range(10, 520, 20)
    trace_ratios = Scatter(x=_range, y=ratios)
    traces = [trace_ratios]
    print traces
    layout = Layout(title="Ratio of soccer versus non-soccer tweets in largest cluster",
                    xaxis=dict(title="K"), yaxis=dict(title="Ratio of correct versus incorrect tweets"))
    fig = Figure(data=traces, layout=layout)
    plot(fig)

    import time
    time.sleep(2)
    trace_max = Scatter(x=_range, y=max)
    traces = [trace_max]
    layout = Layout(title="Amount of soccer tweets in largest cluster",
                    xaxis=dict(title="K"),
                    yaxis=dict(title="Amount of datapoints"))

    fig = Figure(data=traces, layout=layout)
    plot(fig)


def plot_groupsizes():
    range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    voetbal = [2, 24, 102, 217, 329, 563, 1162, 1966, 2997, 2753, 2065, 851, 364, 119, 49,27,17,18,1,1]
    jihad = [7, 34, 70, 67, 125, 129,205,246, 211, 172, 92, 98, 46, 27, 7, 2, 1]
    trace_voetbal = Scatter(x=range, y=voetbal, name="voetbal")
    import numpy as np
    print np.array(voetbal).sum()
    print np.array(jihad).sum()
    trace_jihad = Scatter(x=range, y=jihad, name="jihad")
    traces = [trace_voetbal, trace_jihad]
    layout = Layout(title="Number of tweets in ntoken groups",
                    xaxis=dict(title="Ntokens group", autotick=False),
                    yaxis=dict(title="Number of tweets"))
    fig = Figure(data=traces, layout=layout)
    # plot(fig)



def test():

    all_data = pd.DataFrame()
    for dir in directories:
        path = result_path + dir + "/thresholds/all_tokens_annotate.csv"
        print path
        data = pd.read_csv(path)
        all_data = all_data.append(data)
    all_data = all_data.sort_values("0", ascending=False)
    grouped = all_data["0"].groupby(all_data["label"])
    _10_group_0 = grouped.get_group(0).sort_values().values.tolist()
    _10_group_1 = grouped.get_group(1).sort_values().values.tolist()
    print len(_10_group_0), len(_10_group_1)
    trace_correct = Scatter(x=range(len(_10_group_1)), y=_10_group_1, mode="markers", marker=dict(color="rgb(0,255,0)"), name="Correct")
    trace_incorrect = Scatter(x=range(len(_10_group_0)), y=_10_group_0, mode="markers", marker=dict(color="rgb(255,0,0)"), name="Incorrect")
    # trace_correct = Histogram(x=_10_group_1, name="Correct", )
    # trace_incorrect = Histogram(x=_10_group_0, name="Incorrect")

    data = [trace_correct, trace_incorrect]
    layout = Layout(title="Tweets about voetbal which are classified correctly or <br> incorrectly using the Binary Search method.", xaxis=dict(title="Tweet id"),
                    yaxis=dict(title="Probabilities from the word2vec model"))
    fig = Figure(data=data, layout=layout)
    # plot(fig)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    print _10_group_1
    y, binEdges= np.histogram(_10_group_1, bins=20)
    bincenters = 0.00001 + 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, color="green", label="Correct")

    y, binEdges = np.histogram(_10_group_0, bins=20)
    from scipy.interpolate import spline
    bincenters = 0.00001 + 0.5 * (binEdges[1:] + binEdges[:-1])
    plt.plot(bincenters, y, color="red", label="Incorrect")

    sns.plt.xticks(fontsize=16)
    sns.plt.yticks(fontsize=16)
    plt.ticklabel_format(useOffset=False)
    plt.title("Distribution of softmax values for all tokens method", fontsize=30)
    plt.xlabel("Hierarchical softmax value", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
    #
    #
    # import scipy.stats as stats
    #
    # noise = np.random.normal(0, 1, (1000,))
    # density = stats.gaussian_kde(_10_group_1)
    # n, x, _ = plt.hist(_10_group_1, histtype='step')
    # plt.plot(x, density(x))
    # plt.show()

test()

#
# import time
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
# prec_recall_thresholds()
# time.sleep(2)

# plot_train_test()
# time.sleep(2)

# plot_loss()
# time.sleep(2)

# plot_f1()
# heads_compared_to_alltokens_using_ntokens()
# compare_all_tokens_vs_ntokens
# plot_cluster_member_ratio_and_max()
# plot_groupsizes()
# from scipy import stats
# stats.kstest(x, 'uniform')