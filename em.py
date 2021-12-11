import numpy as np
import sys
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from reader import Reader
import math


class EMGMM:
	def __init__(self, train_data, test_data, train_data_male, train_data_female, test_data_male, test_data_female):
		self.train_data = train_data
		self.test_data = test_data
		self.train_data_male = train_data_male
		self.train_data_female = train_data_female
		self.test_data_male = test_data_male
		self.test_data_female = test_data_female

	def run(self, n_components, n_init=1, cov_type="full", gender="all"):
		self.n_components = list(n_components)
		self.means = [None]*10
		self.covariances = [None]*10
		self.weights = [None]*10

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
			gmm = GaussianMixture(n_components=self.n_components[digit], n_init=n_init, covariance_type=cov_type, random_state=0)
			gfit = gmm.fit(train_data[digit])
			#print(gfit.converged_)
			self.means[digit] = gfit.means_
			self.weights[digit] = gfit.weights_
			self.covariances[digit] = gfit.covariances_

		likelihoods = np.zeros([10, num_blocks, 10])
		classifications = np.zeros([10,num_blocks])
		for class_digit in range(10):
			for data_digit in range(10):
				for j, block in enumerate(test_data[data_digit]):
					likelihoods[data_digit][j][class_digit] = self._ml_classification_for_digit(block, 
						class_digit, self.n_components[digit])

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
	e = EMGMM(r.train_data_digits, r.test_data_blocks, 
		r.train_data_digits_male, r.train_data_digits_female,
		r.test_data_blocks_male, r.test_data_blocks_female)
	const_phonemes = [6]*10 # Constant number of clusters
	phonemes = [4, 4, 5, 5, 5, 4, 4, 4, 5, 4] # Phonemes for digits 0 through 9
	phonemes_transitions = [2*p for p in phonemes] # Phonemes + Transitions
	print("Results for Full Covariance with 5 Clusters:")
	e.run(n_components=const_phonemes, cov_type="full")
	#print("Results for Full Covariance with Clusters = Phonemes:")
	#e.run(n_components=phonemes, cov_type="full")
	#print("Results for Full Covariance with Clusters = Phonemes + Transitions:")
	#e.run(n_components=phonemes_transitions, cov_type="full")

if __name__ == "__main__":
	main()



