import numpy as np
import sys
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from reader import Reader
import math


class EMGMM:
	def __init__(self, train_data, test_data):
		self.train_data = train_data
		self.test_data = test_data

	def run(self, n_components, n_init=1):
		self.n_components = n_components
		self.gmm = GaussianMixture(n_components=self.n_components, n_init=n_init, random_state=0)

		self.means = np.empty([10, self.n_components, 13])
		self.covariances = np.empty([10, self.n_components, 13, 13])
		self.weights = np.empty([10, self.n_components])

		for digit in range(10):
			gfit = self.gmm.fit(self.train_data[digit])
			print(gfit.converged_)
			self.means[digit] = gfit.means_
			self.weights[digit] = gfit.weights_
			self.covariances[digit] = gfit.covariances_

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

	def _ml_classification_for_digit(self, block, digit):
		# Using formula from project guidance
		likelihood = 1
		pi = np.zeros([self.n_components])
		for i, frame in enumerate(block):
			total = 0
			for m in range(self.n_components):
				pdf = multivariate_normal.pdf(frame, 
					mean=self.means[digit][m], 
					cov=self.covariances[digit][m])
				total += (self.weights[digit][m] * pdf)
			likelihood *= total
		return likelihood


def main():
	#np.set_printoptions(threshold=sys.maxsize)
	r = Reader()
	r.read()
	e = EMGMM(r.train_data_digits, r.test_data_blocks)
	e.run(n_components=5)

if __name__ == "__main__":
	main()



