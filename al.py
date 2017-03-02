import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def find_threshold_subject(hashtag, root):
	import pandas as pd

	# print "The subject is: %s" % hashtag
	# print "Please press 1 if the tweets are about the subject or press 0 otherwise. "

	data = pd.read_csv(root + "results/probs/" + hashtag + ".csv", usecols=["id", "probs", "text"])
	data = data.dropna()

	data["probs"] = data["probs"].apply(lambda x: float(x))
	data = data.sort_values(by=["probs"], ascending=False) # changed sort to sort_values and the parameter colums= to by=
	probs = data["probs"]
	start_t = 1.0  # (max(probs)-min(probs))/2
	delta = 0.0001
	# tmp = data[data["probs"] > t]
	# print tmp.head()["text"].values
	# if probs are large enough to be voetbal, lower t.

	first = True # some subjects have tweets with the highest probability which are not about the subject
	round_count = 0
	# assume the list is ordered from voetbal -2 to not voetbal +10 or sth.
	while True:
		print "The subject is: %s" % hashtag
		print "Please press 1 if the tweets are about the subject or press 0 otherwise. "
		logger.info("The start threshold is: %f" % start_t)
		tmp = data[data["probs"] < start_t]
		print tmp.head()["text"].values
		flag = True
		inp = int(input())
		t = start_t
		prev_t = start_t
		while flag:
			logger.info("The current threshold is %f", t)
			if inp == 1:
				# print "about the subject"
				t -=delta
				tmp = data[data["probs"] < t]
				print tmp.head()["text"].values
				try:
					inp = int(input())
				except (SyntaxError, NameError):
					print "wrong input, please press 1 or 0"
					inp = int(input())
			if inp == 0:
				if first:
					t -= 0.1
				# print "not about the subject!"
				flag = False
				prev_t = t+delta
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
			tmp = data[data["probs"] < center]
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
			if inp == 0:
				new_low = high-(high-center)/2.0
				return high, center, high-(high-center)/2.0
			elif inp == 1:
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
	logger.info("Save threshold in /thresholds/ for %s", hashtag)
	f = open(root + "results/thresholds/" + hashtag, "w")
	f.write(str(center))
	f.close()

def get_threshold_subject(path, hashtag):
	logger.info("Returning threshold for subject: %s", hashtag)
	threshold = open(path + hashtag, "r").readline()
	return  float(threshold)