def find_threshold_subject(hashtag):
	import pandas as pd
	data = pd.read_hdf("/media/cluster/data1/lambert/results/probs/" + hashtag + ".h5")["data"]

	data = data.sort(columns=["probs"])
	probs = data[2]
	t = 0#(max(probs)-min(probs))/2
	tmp = data[data[2] > t]
	print tmp.head()[3].values
	# if probs are large enough to be voetbal, lower t.


	# assume the list is ordered from voetbal -2 to not voetbal +10 or sth.
	while True:
		tmp = data[data[2] > t]
		print tmp.head()[3].values
		flag = True
		inp = int(input())
		t = 0
		while flag:
			print t
			if inp == 0:
				print "not about the subject"
				t +=1
				tmp = data[data[2] > t]
				print tmp.head()[3].values
				inp = int(input())
			if inp == 1:
				print "about the subject!"
				flag = False
				prev_t = t-1
		print "the threshold is %i" % t
		print "the previous threshold was %i" % prev_t
		high = t

		low = prev_t
		center = high-(high-low)/2.0

		def test(high,low, center):
			tmp = data[data[2] > center]
			print tmp.head()[3].values
			inp = int(input())
			if inp == 1:
				new_low = high-(high-center)/2.0
				return high, new_low, high-(center-new_low)/2.0
			elif inp == 0:
				return center, low, center - (center - low)/2.0

		while True:
			high, low, center = test(high, low, center)
			print high, low, center




		while True:

			if inp == 1:
				t = t - ((t-prev_t) / 2.0)
				print "the new threshold is %f" % t
				tmp = data[data[2] > t]
				print tmp.head()[3].values

			elif inp == 0:
				t = t + ((t-prev_t) / 2.0)
				print "the new threshold is %f" % t
				tmp = data[data[2] > t]
				print tmp.head()[3].values
			inp = int(input())

		f = open("/media/cluster/data1/lambert/results/thresholds/" + hashtag, "w")
		f.write(str(t))
		f.close()

def get_threshold_subject(path):
	threshold = float(open(path, "r").readline())