# -*- coding: utf-8 -*-
# __author__: Yixuan LI
# __email__: yl2363@cornell.edu

import numpy as np
from scipy.stats import ttest_ind
import os



class PearsonData:
	def __init__(self,inputDir_geo,inputDir_non_geo,outputDir):
		self.inputDir_geo = inputDir_geo
		self.inputDir_non_geo = inputDir_non_geo
		self.outputDir = outputDir
		#os.system("mkdir -p %s"%(outputDir))

		self.lineCount  = 1
		self.IDcount = 0
		self.id = []
		#self.maxLine = 118
		self.maxLine = 545
		self.LIWC_attribute = 68
		self.data_geo = np.zeros((self.maxLine,self.LIWC_attribute))
		self.data_non_geo = np.zeros((self.maxLine,self.LIWC_attribute))
		self.LIWC = None

	def read_data_geo(self):
		with open(self.inputDir_geo,'r') as fin:
			indexLine = 0
			for line in fin:
				print line
				if self.lineCount == 1:
					self.LIWC = line.split('\t')[1:][:-1]
					print self.LIWC
				if self.lineCount > 1:    # skip the header
					values = line.split('\t')[1:][:-1]
					print values
					values = [float(s) for s in values]   # convert string to float
					
					
					self.data_geo[indexLine,:] = values
					indexLine += 1
					if indexLine == self.maxLine:
						break
					print indexLine
				self.lineCount += 1
		#print self.data_geo
	
	def read_data_non_geo(self):
		self.lineCount = 1
		with open(self.inputDir_non_geo,'r') as fin:
			indexLine = 0
			for line in fin:
				if self.lineCount > 1:    # skip the header
					values = line.split('\t')[1:][:-1]
					print values
					values = [float(s) for s in values]   # convert string to float
					
					self.data_non_geo[indexLine,:] = values
					indexLine += 1
					if indexLine == self.maxLine:
						break
					print indexLine
				self.lineCount += 1
	
	
	def cal_pearson(self):
		if os.path.exists(self.outputDir):
			os.remove(self.outputDir)
		with open(self.outputDir,'a') as fout:
			fout.write("LIWC_attribute"+'\t'+"t-statistic"+'\t'+"p_value\n")
		for i in range(self.LIWC_attribute):
			vec1 = self.data_geo[:,i]
			vec2 = self.data_non_geo[:,i]
			#print vec1
			#print vec2
			t,pval = ttest_ind(vec1,vec2)
			with open(self.outputDir,'a') as fout:
				string = self.LIWC[i] +'\t'+str(t)+'\t'+str(pval)+'\n'
				fout.write(string)


if __name__=='__main__':


#########################################################################################
	inputDir_geo = '../txt/geo/3_13_threshold600_complete_803users.txt'
	#inputDir_geo = '../txt/geo/3_12_991users.txt'
	inputDir_geo = '../../LIWC/txt/geo/3_18_threshold600_manual_filtered_765users.txt'
	inputDir_non_geo = '../txt/non_geo/3_12_865users.txt'
	inputDir_non_geo = '../txt/non_geo/3_24_945users.txt'
	inputDir_non_geo = '../txt/non_geo/3_25_545users.txt'
	outputDir = 'pvalues_threshold_manual_filter_both_545.txt'
	calculator = PearsonData(inputDir_geo,inputDir_non_geo,outputDir)
	#generator.write_tsv()
	calculator.read_data_geo()
	calculator.read_data_non_geo()
	calculator.cal_pearson()



