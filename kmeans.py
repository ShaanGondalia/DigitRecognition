import numpy as np
import sys
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from reader import Reader
import math


class KMeansGMM:
	def __init__(self, train_data, test_data):
		self.train_data = train_data
		self.test_data = test_data

	def run(self, n_phonemes, n_init=10):
		self.n_clusters = n_phonemes
		self.kmeans = KMeans(init="k-means++", n_clusters=self.n_clusters, n_init=n_init, random_state=0)
		self.cluster_centers = np.empty([10, self.n_clusters, 13])
		self.covariances = np.empty([10, self.n_clusters, 13, 13])
		self.counts = np.zeros([10, self.n_clusters])
		self.total_frames = np.zeros([10])

		for digit in range(10):
			kfit = self.kmeans.fit(self.train_data[digit])
			self.cluster_centers[digit] = kfit.cluster_centers_
			labels = kfit.labels_
			self.covariances[digit] = self._compute_cov(digit, labels)
			for label in labels:
				self.counts[digit][label] += 1
				self.total_frames[digit] += 1

		self.likelihoods = np.zeros([10, 220, 10])
		self.classifications = np.zeros([10,220])
		for class_digit in range(10):
			for data_digit in range(10):
				for j, block in enumerate(self.test_data[data_digit]):
					self.likelihoods[data_digit][j][class_digit] = self._ml_classification_for_digit(block, class_digit)

		correct = 0
		for digit in range(10):
			for block in range(220):
				self.classifications[digit][block] = np.argmax(self.likelihoods[digit][block])
				if(self.classifications[digit][block] == digit):
					correct+=1

		print(self.classifications[0])
		print("Accuracy: {}".format(correct/2200))

	def _compute_cov(self, digit, labels):
		cov = np.empty([self.n_clusters, 13, 13])
		clustered_data = [[]]*self.n_clusters
		for i, frame in enumerate(self.train_data[digit]):
			classification = labels[i]
			clustered_data[classification].append(frame)
		for i, cluster in enumerate(clustered_data):
			cov[i] = np.cov(cluster, rowvar=False)
		return cov

	def _ml_classification_for_digit(self, block, digit):
		# Using formula from project guidance
		likelihood = 1
		pi = np.zeros([self.n_clusters])
		for m in range(self.n_clusters):
			pi[m] = self.counts[digit][m]/self.total_frames[digit]

		for i, frame in enumerate(block):
			total = 0
			for m in range(self.n_clusters):
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
	k = KMeansGMM(r.train_data_digits, r.test_data_blocks)
	k.run(n_phonemes=5)

if __name__ == "__main__":
	main()



