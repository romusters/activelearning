import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def find_threshold_subject(hashtag, root):
    import pandas as pd

    # print "The subject is: %s" % hashtag
    # print "Please press 1 if the tweets are about the subject or press 0 otherwise. "

    data = pd.read_csv(root + "results/probs/" + hashtag + ".csv", usecols=["id", "0", "text"])
    data = data.dropna()

    data["0"] = data["0"].apply(lambda x: float(x))
    data = data.sort_values(by=["0"], ascending=False) # changed sort to sort_values and the parameter colums= to by=
    probs = data["0"]
    # start_t = 1.0  # (max(probs)-min(probs))/2
    start_t = data["0"][0]
    print start_t
    delta = 0.01
    # tmp = data[data["probs"] > t]
    # print tmp.head()["text"].values
    # if probs are large enough to be voetbal, lower t.

    first = True # some subjects have tweets with the highest probability which are not about the subject
    round_count = 0
    labels = []
    ids = []
    tweets = []
    # assume the list is ordered from voetbal -2 to not voetbal +10 or sth.
    while True:
        print "The subject is: %s" % hashtag
        print "Please press 1 if the tweets are about the subject or press 0 otherwise. "
        logger.info("The start threshold is: %f" % start_t)
        tmp = data[data["0"] < start_t]
        head = tmp.head(1)
        tweet =  head["text"].values[0]
        tweets.append(tweet)
        id = head["id"].values[0]
        ids.append(id)
        print tweet
        flag = True
        inp = int(input())

        t = start_t
        prev_t = start_t
        if inp == 1:
            t-=delta
            prev_t = t
            # labels.append(1)


        while flag:
            logger.info("The current threshold is %f", t)
            if inp == 1:
                # print "about the subject"
                t -=delta
                tmp = data[data["0"] < t]
                head = tmp.head(1)
                tweet = head["text"].values[0]
                tweets.append(tweet)
                id = head["id"].values[0]
                ids.append(id)
                print tweet
                try:
                    inp = int(input())
                except (SyntaxError, NameError):
                    print "wrong input, please press 1 or 0"
                    inp = int(input())
                labels.append(1)
            if inp == 0:
                if first:
                    t = prev_t
                # print "not about the subject!"
                flag = False
                prev_t = t+delta
                labels.append(0)
            if round_count > 10:
                flag = False
            round_count+=1
        center = None

        if round_count > 10:
            logger.info("Max iterations reached.")
            center = t
            break
        logger.info("The threshold is %f", t)
        logger.info("the previous threshold was %f", prev_t)
        low = t
        high = prev_t
        center = high-(high-low)/2.0

        def test(high,low, center):
            print "The subject is: %s" % hashtag
            print "Please press 1 if the tweets are about the subject or press 0 otherwise. "
            tmp = data[data["0"] < center]
            head = tmp.head(1)
            tweet = head["text"].values[0]
            tweets.append(tweet)
            id = head["id"].values[0]
            ids.append(id)
            print tweet
            # If there is no data above the threshold
            if len(tmp.index) == 0:
                print "empty DATA"
                return -1,-1,-1
            try:
                inp = int(input())
            except (SyntaxError, NameError):
                print "wrong input, please press 1 or 0"
                inp = int(input())
            if inp == 0:
                new_low = high-(high-center)/2.0
                labels.append(0)
                return high, center, high-(high-center)/2.0
            elif inp == 1:
                labels.append(1)
                return center, low, center - (center - low)/2.0

        prev_center = 1.0
        while True:
            logger.info("The current threshold is %f", center)
            high, low, center = test(high, low, center)
            print high, low, center
            print prev_center, center
            if abs(prev_center-center) < 0.00000001:

                logger.info("Difference very small, stopping.")
                break
            if round_count > 10:
                break
            round_count +=1
            prev_center = center
        break
    result = pd.DataFrame({"ids": ids, "labels": labels, "text": tweets})
    result.to_csv(root + "results/thresholds/al_annotation_"+ hashtag + ".csv")
    print labels
    print ids
    print tweets
    logger.info("Save threshold in /thresholds/ for %s", hashtag)
    f = open(root + "results/thresholds/" + hashtag, "w")
    f.write(str(center))
    f.close()

def get_threshold_subject(path, hashtag):
    logger.info("Returning threshold for subject: %s", hashtag)
    threshold = open(path + hashtag, "r").readline()
    return  float(threshold)

def n_tokens_accuracy(root, hashtag):
    import pandas as pd
    print "press 0 if tweet is not about voetbal or 1 if it does."
    acc = []
    ntokens_list = range(1,20)
    for i in ntokens_list:
        tmp_acc = 0
        data = pd.read_csv(root + "/results/test/voetbal/voetbal" + str(i) + ".csv")
        top = data.head(20)
        tweets = top.text.values.tolist()
        print tweets
        for tweet in tweets:
            print tweet
            res = int(input())
            tmp_acc += res
        acc.append(tmp_acc)
    df = pd.DataFrame({"ntokens":ntokens_list, "acc": acc})
    result_path = root + "/results/test/voetbal/result.csv"
    df.to_csv(result_path)

find_threshold_subject("10", "/home/robert/lambert/")
find_threshold_subject("all_tokens", "/home/robert/lambert/")