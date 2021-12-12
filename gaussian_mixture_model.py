import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from reader import Reader
import math
import time
import seaborn as sns
import matplotlib.pyplot as plt


class GaussianMixtureModel:
	def __init__(self, train_data, test_data, train_data_male, train_data_female, test_data_male, test_data_female):
		self.train_data = train_data
		self.test_data = test_data
		self.train_data_male = train_data_male
		self.train_data_female = train_data_female
		self.test_data_male = test_data_male
		self.test_data_female = test_data_female

	def run(self, n_components, title, gmm_type="kmeans", cov_type="full", gender="all", mfcc_count=13):
		self.n_components = list(n_components)
		self.mfcc_count = mfcc_count
		self.means = [None]*10
		self.covariances = [None]*10
		self.weights = [None]*10
		self.counts = [None]*10
		self.total_frames = np.zeros([10])

		# Seperate Training/Testing data based on gender
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

		# Prune mfccs based on mfcc count (Dimensionality Redcution)
		pruned_train_data = []
		pruned_test_data = []
		for digit in range(10):
			pruned_train_data.append([])
			pruned_test_data.append([])
			for frame in train_data[digit]:
				pruned_train_data[digit].append(frame[0:mfcc_count])
			for i, block in enumerate(test_data[digit]):
				pruned_test_data[digit].append([])
				for frame in block:
					pruned_test_data[digit][i].append(frame[0:mfcc_count])
		train_data = pruned_train_data
		test_data = pruned_test_data

		for digit in range(10):
			if gmm_type=="k-Means":
				kmeans = KMeans(init="k-means++", n_clusters=self.n_components[digit], n_init=10, random_state=0)
				kfit = kmeans.fit(train_data[digit])
				labels = kfit.labels_
				self.means[digit] = kfit.cluster_centers_
				self.covariances[digit] = self._compute_cov(digit, labels, self.n_components[digit], train_data, cov_type)
				self.counts[digit] = np.zeros([self.n_components[digit]])
				for label in labels:
					self.counts[digit][label] += 1
					self.total_frames[digit] += 1
				self.weights[digit] = np.zeros([self.n_components[digit]])
				for m in range(self.n_components[digit]):
					self.weights[digit][m] = self.counts[digit][m]/self.total_frames[digit]

			else:
				gmm = GaussianMixture(n_components=self.n_components[digit], n_init=1, covariance_type=cov_type, random_state=0)
				gfit = gmm.fit(train_data[digit])
				#print(gfit.converged_)
				self.means[digit] = gfit.means_
				self.weights[digit] = gfit.weights_
				self.covariances[digit] = gfit.covariances_

		likelihoods = np.zeros([10, num_blocks, 10])
		classifications = np.zeros([10, num_blocks])
		for class_digit in range(10):
			for data_digit in range(10):
				for j, block in enumerate(test_data[data_digit]):
					likelihoods[data_digit][j][class_digit] = self._ml_classification_for_digit(block, 
						self.weights[data_digit], class_digit, self.n_components[digit])

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
		self._plot_confusion_matrix(confusion_matrix)

	def _compute_cov(self, digit, labels, clusters, train_data, cov_type):
		cov = np.empty([clusters, self.mfcc_count, self.mfcc_count])
		clustered_data = [[]]*clusters
		for i, frame in enumerate(train_data[digit]):
			classification = labels[i]
			clustered_data[classification].append(frame)
		for i, cluster in enumerate(clustered_data):
			cov[i] = np.cov(cluster, rowvar=False)

		if cov_type != "full":
			diag = np.zeros([clusters, self.mfcc_count, self.mfcc_count])
			spherical = np.zeros([clusters, self.mfcc_count, self.mfcc_count])
			for i in range(clusters):
				diag[i]=np.diag(np.diag(cov[i]))
				spherical[i] = np.identity(self.mfcc_count)*np.linalg.det(diag[i])
			if cov_type=="spherical":
				return spherical
			return diag
		return cov

	def _ml_classification_for_digit(self, block, weights, digit, clusters):
		# Using formula from project guidance
		likelihood = 1
		for i, frame in enumerate(block):
			total = 0
			for m in range(clusters):
				pdf = multivariate_normal.pdf(frame, 
					mean=self.means[digit][m], 
					cov=self.covariances[digit][m])
				total += (weights[m] * pdf)
			likelihood *= total
		return likelihood

	def _plot_confusion_matrix(self, cf_matrix, title):
		ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)

		ax.set_title(title);
		ax.set_xlabel('Predicted Digits')
		ax.set_ylabel('Actual Digits ');

		## Ticket labels - List must be in alphabetical order
		ax.xaxis.set_ticklabels([0,1,2,3,4,5,6,7,8,9])
		ax.yaxis.set_ticklabels([0,1,2,3,4,5,6,7,8,9])

		## Display the visualization of the Confusion Matrix.
		plt.show()


def main():
	#np.set_printoptions(threshold=sys.maxsize)
	r = Reader()
	r.read()
	m = GaussianMixtureModel(r.train_data_digits, r.test_data_blocks, 
		r.train_data_digits_male, r.train_data_digits_female,
		r.test_data_blocks_male, r.test_data_blocks_female)
	const_phonemes = [5]*10 # Constant number of clusters
	phonemes = [4, 4, 5, 5, 5, 4, 4, 4, 5, 4] # Phonemes for digits 0 through 9
	phonemes_transitions = [2*p for p in phonemes] # Phonemes + Transitions
	plt.ion()

	print("k-Means Results for 5 Clusters:")
	m.run(const_phonemes, "k-Means with 5 Clusters", gmm_type="k-Means", cov_type="full", gender="all", mfcc_count=13)
	print("k-Means Results for Clusters = Phonemes:")
	m.run(phonemes, "k-Means with Clusters = Phonemes", gmm_type="k-Means", cov_type="full", gender="all", mfcc_count=13)
	print("k-Means Results for Clusters = Phonemes + Transitions:")
	m.run(phonemes_transitions, "k-Means with Clusters = Phonemes + Transitions", gmm_type="k-Means", cov_type="full", gender="all", mfcc_count=13)

	print("k-Means Results for 5 Clusters, Diagonal Covariance:")
	m.run(const_phonemes, "k-Means with 5 Clusters, Diagonal Covariance", gmm_type="k-Means", cov_type="diag", gender="all", mfcc_count=13)

	print("k-Means Results for 5 Clusters, Spherical Covariance:")
	m.run(const_phonemes, "k-Means with 5 Clusters, Spherical Covariance", gmm_type="k-Means", cov_type="spherical", gender="all", mfcc_count=13)

	print("k-Means Results for 5 Clusters, Male:")
	m.run(const_phonemes, "k-Means with 5 Clusters, Male", gmm_type="k-Means", cov_type="full", gender="male", mfcc_count=13)
	print("k-Means Results for 5 Clusters, Female:")
	m.run(const_phonemes, "k-Means with 5 Clusters, Female", gmm_type="k-Means", cov_type="full", gender="female", mfcc_count=13)

	print("k-Means Results for 5 Clusters, 5 MFCCs:")
	m.run(const_phonemes, "k-Means with 5 Clusters, 5 MFCCs", gmm_type="k-Means", cov_type="full", gender="all", mfcc_count=5)
	print("k-Means Results for 5 Clusters, 9 MFCCs:")
	m.run(const_phonemes, "k-Means with 5 Clusters, 9 MFCCs", gmm_type="k-Means", cov_type="full", gender="all", mfcc_count=9)

	print("EM Results for 5 Clusters:")
	m.run(const_phonemes, "EM with 5 Clusters", gmm_type="EM", cov_type="full", gender="all", mfcc_count=13)
	print("EM Results for Clusters = Phonemes:")
	m.run(phonemes, "EM with Clusters = Phonemes", gmm_type="EM", cov_type="full", gender="all", mfcc_count=13)
	print("EM Results for Clusters = Phonemes + Transitions:")
	m.run(phonemes_transitions, "EM with Clusters = Phonemes + Transitions", gmm_type="EM", cov_type="full", gender="all", mfcc_count=13)

	print("EM Results for 5 Clusters, Diagonal Covariance:")
	m.run(const_phonemes, "EM with 5 Clusters, Diagonal Covariance", gmm_type="EM", cov_type="diag", gender="all", mfcc_count=13)

	print("EM Results for 5 Clusters, Spherical Covariance:")
	m.run(const_phonemes, "EM with 5 Clusters, Spherical Covariance", gmm_type="EM", cov_type="spherical", gender="all", mfcc_count=13)

	print("EM Results for 5 Clusters, Male:")
	m.run(const_phonemes, "EM with 5 Clusters, Male", gmm_type="EM", cov_type="full", gender="male", mfcc_count=13)
	print("EM Results for 5 Clusters, Female:")
	m.run(const_phonemes, "EM with 5 Clusters, Female", gmm_type="EM", cov_type="full", gender="female", mfcc_count=13)

	print("EM Results for 5 Clusters, 5 MFCCs:")
	m.run(const_phonemes, "EM with 5 Clusters, 5 MFCCs", gmm_type="EM", cov_type="full", gender="all", mfcc_count=5)
	print("EM Results for 5 Clusters, 9 MFCCs:")
	m.run(const_phonemes, "EM with 5 Clusters, 9 MFCCs", gmm_type="EM", cov_type="full", gender="all", mfcc_count=9)


if __name__ == "__main__":
	main()



