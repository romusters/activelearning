import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def find_threshold_subject(hashtag):
	import pandas as pd

	# print "The subject is: %s" % hashtag
	# print "Please press 1 if the tweets are about the subject or press 0 otherwise. "

	data = pd.read_csv("/media/cluster/data1/lambert/results/probs/" + hashtag + ".csv", usecols=["id", "probs", "text"])
	data = data.dropna()

	data["probs"] = data["probs"].apply(lambda x: float(x))
	data = data.sort_values(by=["probs"]) # changed sort to sort_values and the parameter colums= to by=
	probs = data["probs"]
	t = 0  # (max(probs)-min(probs))/2
	# tmp = data[data["probs"] > t]
	# print tmp.head()["text"].values
	# if probs are large enough to be voetbal, lower t.

	round_count = 0
	# assume the list is ordered from voetbal -2 to not voetbal +10 or sth.
	while True:
		print "The subject is: %s" % hashtag
		print "Please press 1 if the tweets are about the subject or press 0 otherwise. "
		tmp = data[data["probs"] > t]
		print tmp.head()["text"].values
		flag = True
		inp = int(input())
		t = 0
		while flag:
			logger.info("The current threshold is %f", t)
			if inp == 0:
				# print "not about the subject"
				t +=1
				tmp = data[data["probs"] > t]
				print tmp.head()["text"].values
				try:
					inp = int(input())
				except (SyntaxError, NameError):
					print "wrong input, please press 1 or 0"
					inp = int(input())
			if inp == 1:
				# print "about the subject!"
				flag = False
				prev_t = t-1
			if round_count > 10:
				flag = False
			round_count+=1
		center = None

		if round_count > 10:
			center = t
			break
		logger.info("The threshold is %i", t)
		logger.info("the previous threshold was %i", prev_t)
		high = t
		low = prev_t
		center = high-(high-low)/2.0

		def test(high,low, center):
			print "The subject is: %s" % hashtag
			print "Please press 1 if the tweets are about the subject or press 0 otherwise. "
			tmp = data[data["probs"] > center]
			print tmp.head()["text"].values
			# If there is no data above the threshold
			if len(tmp.index) == 0:
				print "empty DATA"
				return -1,-1,-1
			try:
				inp = int(input())
			except (SyntaxError, NameError):
				print "wrong input, please press 1 or 0"
				inp = int(input())
			if inp == 1:
				new_low = high-(high-center)/2.0
				return high, new_low, high-(center-new_low)/2.0
			elif inp == 0:
				return center, low, center - (center - low)/2.0

		prev_center = 99999
		while True:
			logger.info("The current threshold is %f", center)
			high, low, center = test(high, low, center)
			print high, low, center

			if prev_center-center < 0.01:
				break
			if round_count > 10:
				break
			round_count +=1
			prev_center = center
		break
	logger.info("Save threshold in /thresholds/ for %s", hashtag)
	f = open("/media/cluster/data1/lambert/results/thresholds/" + hashtag, "w")
	f.write(str(center))
	f.close()

def get_threshold_subject(path, hashtag):
	logger.info("Returning threshold for subject: %s", hashtag)
	threshold = open(path + hashtag, "r").readline()
	return  float(threshold)