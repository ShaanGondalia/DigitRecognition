import matplotlib.pyplot as plt
import numpy as np


class Reader:

	TEST_FILENAME = "./data/Test_Arabic_Digit.txt"
	TRAIN_FILENAME = "./data/Train_Arabic_Digit.txt"
	MFCC_COLORS = ['black', 'red', 'orange', 'gold', 
				   'lawngreen', 'darkgreen', 'cyan',
				   'dodgerblue', 'navy', 'violet',
				   'deeppink', 'olive', 'slategray']

	def __init__(self):
		self.test_data = {}
		self.train_data = {}
		self.train_mfccs = {}
		self.test_mfccs = {}
		self.train_data_digits = []
		self.test_data_blocks = []
		self.train_data_digits_male = []
		self.train_data_digits_female = []
		self.test_data_blocks_male = []
		self.test_data_blocks_female = []

	def read(self):
		# Read Test data from file
		with open(self.TEST_FILENAME) as file:
			lines = file.readlines()
			self.test_data = self.parse_lines(lines)

		# Read Training data from file
		with open(self.TRAIN_FILENAME) as file:
			lines = file.readlines()
			self.train_data = self.parse_lines(lines)

		self.train_mfccs = self.convert_MFCCS(self.train_data)
		self.test_mfccs = self.convert_MFCCS(self.test_data)
		self.seperate_digits()
		self.seperate_digits_male_female()

	def parse_lines(self, lines):
		i=0
		blocks = {}
		for line in lines:
			if line == "            \n":
				i+=1
				blocks[i] = []
			else:
				split = line.rstrip().split(" ")
				split = [float(s) for s in split]
				blocks[i].append(split)
		return blocks

	def convert_MFCCS(self, data):
		ret = {}
		for k in data.keys():
			MFCCS = []
			for i in range(13):
				MFCCS.append([])
				for line in data[k]:
					MFCCS[i].append(line[i])
			ret[k] = MFCCS
		return ret

	def plot(self, block):
		plt.title(f'MFCC vs Frame Index - Digit {int((block-1)/660)} Block {block%660}')
		plt.xlabel('Frame Index')
		plt.ylabel('MFCC')
		plt.grid(alpha=.4,linestyle='--')

		for i in range(13):
			plt.plot(range(len(self.train_mfccs[block][i])), self.train_mfccs[block][i],
				c=self.MFCC_COLORS[i], label=f'MFCC{i+1}')

		plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
		plt.subplots_adjust(right=0.8)

		plt.show()

	def seperate_digits(self):
		# Seperates blocks into digits
		for i in range(10):
			self.train_data_digits.append([])
			self.test_data_blocks.append([])
			for j in range(1,661):
				for frame in self.train_data[660*i+j]:
					self.train_data_digits[i].append(frame)
			for j in range(1,221):
				self.test_data_blocks[i].append([])
				# Keep test data seperateed by both digits and blocks, necessary for ML Classification
				self.test_data_blocks[i][j-1] = list(self.test_data[220*i+j])

	def seperate_digits_male_female(self):
		for i in range(10):
			self.train_data_digits_male.append([])
			self.train_data_digits_female.append([])
			self.test_data_blocks_male.append([])
			self.test_data_blocks_female.append([])
			for j in range(1, 331):
				for frame in self.train_data[660*i+j]:
					self.train_data_digits_male[i].append(frame)
				for frame in self.train_data[660*i+j+330]:
					self.train_data_digits_female[i].append(frame)
			for j in range(0,110):
				self.test_data_blocks_male[i].append([])
				self.test_data_blocks_female[i].append([])
				self.test_data_blocks_male[i][j] = list(self.test_data_blocks[i][j])
				self.test_data_blocks_female[i][j] = list(self.test_data_blocks[i][j+110])

	def plot_gender(self, digit, mfcc):
		plt.title(f'MFCC {mfcc} vs Normalized Frame Index - Digit {digit}')
		plt.xlabel('Normalized Frame Index (0 to 1)')
		plt.ylabel(f'MFCC {mfcc}')
		plt.grid(alpha=.4,linestyle='--')

		male_normalized_frame_index=[]
		male_data=[]
		for i in range(1,101):
			frame_length = len(self.train_mfccs[digit*660+i][mfcc-1])
			for j, f in enumerate(self.train_mfccs[digit*660+i][mfcc-1]):
				male_normalized_frame_index.append(j/(frame_length-1))
				male_data.append(f)

		female_normalized_frame_index=[]
		female_data=[]
		for i in range(331,431):
			frame_length = len(self.train_mfccs[digit*660+i][mfcc-1])
			for j, f in enumerate(self.train_mfccs[digit*660+i][mfcc-1]):
				female_normalized_frame_index.append(j/(frame_length-1))
				female_data.append(f)

		plt.scatter(male_normalized_frame_index, male_data, c='blue', label='Male', s=5)
		plt.scatter(female_normalized_frame_index, female_data, c='red', label='Female', s=5)

		plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
		plt.subplots_adjust(right=0.8)

		plt.show()
				

def main():
	r = Reader()
	r.read()
	#r.plot(660*0 + 1)
	#print(len(r.train_data_digits_male[0]))
	#print(r.test_data_blocks[9][219])
	#print(r.train_data_digits_male)
	r.plot_gender(2, 1)


if __name__ == "__main__":
	main()

