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

	def run(self, n_components, n_init=1, cov_type="full"):
		self.n_components = list(n_components)
		self.means = [None]*10
		self.covariances = [None]*10
		self.weights = [None]*10

		for digit in range(10):
			gmm = GaussianMixture(n_components=self.n_components[digit], n_init=n_init, covariance_type=cov_type, random_state=0)
			gfit = gmm.fit(self.train_data[digit])
			#print(gfit.converged_)
			self.means[digit] = gfit.means_
			self.weights[digit] = gfit.weights_
			self.covariances[digit] = gfit.covariances_

		likelihoods = np.zeros([10, 220, 10])
		classifications = np.zeros([10,220])
		for class_digit in range(10):
			for data_digit in range(10):
				for j, block in enumerate(self.test_data[data_digit]):
					likelihoods[data_digit][j][class_digit] = self._ml_classification_for_digit(block, 
						class_digit, self.n_components[digit])

		correct = np.zeros([10])
		confusion_matrix = np.zeros([10, 10])
		for digit in range(10):
			for block in range(220):
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
			print(f"    Digit {digit}: {correct[digit]/220}")
		print(f"Total Accuracy: {total_correct/2200}\n")

	def _ml_classification_for_digit(self, block, digit, components):
		# Using formula from project guidance
		likelihood = 1
		pi = np.zeros([components])
		for i, frame in enumerate(block):
			total = 0
			for m in range(components):
				pdf = multivariate_normal.pdf(frame, 
					mean=self.means[digit][m], 
					cov=self.covariances[digit][m])
				total += (self.weights[digit][m] * pdf)
			likelihood *= total
		return likelihood


def main():
	np.set_printoptions(threshold=sys.maxsize)
	r = Reader()
	r.read()
	e = EMGMM(r.train_data_digits, r.test_data_blocks)
	const_phonemes = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] # Constant number of clusters
	phonemes = [4, 4, 5, 5, 6, 4, 4, 4, 6, 3] # Phonemes for digits 0 through 9
	phonemes_transitions = [2*p for p in phonemes] # Phonemes + Transitions
	print("Results for Full Covariance with 5 Clusters:\n")
	e.run(n_components=const_phonemes, cov_type="full")
	print("Results for Full Covariance with Clusters = Phonemes:\n")
	e.run(n_components=phonemes, cov_type="full")
	print("Results for Full Covariance with Clusters = Phonemes + Transitions:\n")
	e.run(n_components=phonemes_transitions, cov_type="full")

if __name__ == "__main__":
	main()



