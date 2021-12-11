import numpy as np
import sys
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from reader import Reader
import math
import time


class KMeansGMM:
	def __init__(self, train_data, test_data, train_data_male, train_data_female, test_data_male, test_data_female):
		self.train_data = train_data
		self.test_data = test_data
		self.train_data_male = train_data_male
		self.train_data_female = train_data_female
		self.test_data_male = test_data_male
		self.test_data_female = test_data_female

	def run(self, n_phonemes, n_init=10, gender="all"):
		self.n_clusters = list(n_phonemes)
		self.cluster_centers = [None]*10
		self.covariances = [None]*10
		self.counts = [None]*10
		self.total_frames = np.zeros([10])

		if gender=="male":
			train_data = self.train_data_male
			test_data = self.test_data_male
			num_blocks = 110
		elif gender=="female":
			train_data = self.train_data_female
			test_data = self.test_data_female
			num_blocks = 110
		else:
			train_data = self.train_data
			test_data = self.test_data
			num_blocks = 220

		for digit in range(10):
			start = time.time()
			kmeans = KMeans(init="k-means++", n_clusters=self.n_clusters[digit], n_init=n_init, random_state=0)
			kfit = kmeans.fit(train_data[digit])
			centers = kfit.cluster_centers_
			self.cluster_centers[digit] = kfit.cluster_centers_
			labels = kfit.labels_

			self.covariances[digit] = self._compute_cov(digit, labels, self.n_clusters[digit])
			self.counts[digit] = np.zeros([self.n_clusters[digit]])
			for label in labels:
				self.counts[digit][label] += 1
				self.total_frames[digit] += 1

		likelihoods = np.zeros([10, num_blocks, 10])
		classifications = np.zeros([10,num_blocks])
		for class_digit in range(10):
			for data_digit in range(10):
				start = time.time()	
				for j, block in enumerate(test_data[data_digit]):
					likelihoods[data_digit][j][class_digit] = self._ml_classification_for_digit(block, 
						class_digit, self.n_clusters[digit])
				end = time.time()
				print(f"Digit blocks: {end-start}")

		correct = np.zeros([10])
		confusion_matrix = np.zeros([10, 10])
		for digit in range(10):
			for block in range(num_blocks):
				classification = np.argmax(likelihoods[digit][block])
				classifications[digit][block] = classification
				confusion_matrix[digit][classification] += 1
				if(classifications[digit][block] == digit):
					correct[digit]+=1

		print("Confusion Matrix: ")
		print(confusion_matrix)
		total_correct = 0;
		print("Accuracy Per Digit: ")
		for digit in range(10):
			total_correct+=correct[digit]
			print(f"    Digit {digit}: {correct[digit]/num_blocks}")
		print(f"Total Accuracy: {total_correct/(num_blocks*10)}\n")

	def _compute_cov(self, digit, labels, clusters):
		cov = np.empty([clusters, 13, 13])
		clustered_data = [[]]*clusters
		for i, frame in enumerate(self.train_data[digit]):
			classification = labels[i]
			clustered_data[classification].append(frame)
		for i, cluster in enumerate(clustered_data):
			cov[i] = np.cov(cluster, rowvar=False)
		return cov

	def _ml_classification_for_digit(self, block, digit, clusters):
		# Using formula from project guidance
		likelihood = 1
		pi = np.zeros([clusters])
		for m in range(clusters):
			pi[m] = self.counts[digit][m]/self.total_frames[digit]

		for i, frame in enumerate(block):
			total = 0
			for m in range(clusters):
				pdf = multivariate_normal.pdf(frame, 
					mean=self.cluster_centers[digit][m], 
					cov=self.covariances[digit][m])
				total += (pi[m] * pdf)
			likelihood *= total
		return likelihood


def main():
	#np.set_printoptions(threshold=sys.maxsize)
	r = Reader()
	r.read()
	k = KMeansGMM(r.train_data_digits, r.test_data_blocks, 
		r.train_data_digits_male, r.train_data_digits_female,
		r.test_data_blocks_male, r.test_data_blocks_female)
	const_phonemes = [6]*10 # Constant number of clusters
	phonemes = [4, 4, 5, 5, 5, 4, 4, 4, 5, 4] # Phonemes for digits 0 through 9
	phonemes_transitions = [2*p for p in phonemes] # Phonemes + Transitions
	print("Results for 5 Clusters:\n")
	k.run(n_phonemes=const_phonemes)
	#print("Results for Clusters = Phonemes:")
	#k.run(n_phonemes=phonemes)
	#print("Results for Clusters = Phonemes + Transitions:")
	#k.run(n_phonemes=phonemes_transitions)

if __name__ == "__main__":
	main()



