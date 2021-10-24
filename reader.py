import matplotlib.pyplot as plt

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
		plt.title('MFCC vs Frame Index - Block {}'.format(block))
		plt.xlabel('Frame')
		plt.ylabel('MFCC')
		plt.grid(alpha=.4,linestyle='--')

		for i in range(13):
			plt.plot(range(len(self.train_mfccs[block][i])), self.train_mfccs[block][i],
				c=self.MFCC_COLORS[i], label='MFCC{}'.format(i+1))

		plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
		plt.subplots_adjust(right=0.8)

		plt.show()

def main():
	r = Reader()
	r.read()
	r.plot(2200)

if __name__ == "__main__":
	main()

