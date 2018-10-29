from __future__ import division
import pandas as pd
import numpy as np
import numpy.linalg as linalg
import math
from svd import SVD,ErrorEstimation

class CUR():
	def __init__(self, file):
		self.filename = file
	
	def dataframe(self):

		"""generates a utility matrix in the form of numpy array"""
		self.utilitymatrix = pd.read_excel(self.filename,headers = None)
		self.utility = self.utilitymatrix.values
		print self.utility

	def probdistribution(self):

		"""generate the probability distribution corresponding to rows and columns"""
		self.sum = 0
		for i in range(0,len(self.utility)):
			for j in range(0,len(self.utility[0,:])):
				self.sum = self.sum + self.utility[i,j]*self.utility[i,j]

		self.rows = []
		self.cols = []

		"""determining probability distribution for randomly 
					selecting rows and columns"""
					
		for i in range(0,len(self.utility)):
			self.r = 0
			for j in range(0,len(self.utility[0,:])):
				self.r = self.r + self.utility[i,j]*self.utility[i,j]

			self.rows.append(self.r/self.sum)

		for i in range(0,len(self.utility[0,:])):
			self.c = 0
			for j in range(0,len(self.utility)):
				self.c = self.c + self.utility[j,i]*self.utility[j,i]

			self.cols.append(self.c/self.sum)

	def randomnumber(self,prob,num,ct,decision_var):
		
		"""generates random numbers lieing in a range with a 
				certain associated probability"""
		if decision_var==1:
			nos = np.random.choice(np.arange(0,num),ct,prob)

		elif decision_var==2:
			nos = np.random.choice(np.arange(0,num),ct,replace=False,p=prob)

		return nos

	def C_calc(self,decision_var):

		"""generates the C matrix by randomly selecting columns of the original matrix by 
					making use of the associated probability distribution"""
		col = self.randomnumber(self.cols,len(self.utility[0,:]),2,decision_var)
		self.c = col
		self.C = self.utility[:,col]
		self.C = self.C.astype(float)
		i = 0
		for j in range(0,2):
			p = self.cols[col[i]]
			i = i+1
			for k in range(0,len(self.utility)):
				num = math.sqrt(2*p)
				self.C[k,j] = (self.C[k,j])/num
		
		print self.C

	def R_calc(self,decision_var):

		"""generates the R matrix by randomly selecting rows of the original matrix by making
		   use of the associated probability distribution which is given by (sum of the square 
		   of the elements of that row)/(sum of the squares of all the elements of the matrix)"""
		
		row = self.randomnumber(self.rows,len(self.utility),2,decision_var)
		self.r = row
		self.R = self.utility[row,:]
		self.R = self.R.astype(float)
		i = 0
		for j in range(0,2):
			p = self.rows[row[i]]
			i = i+1
			for k in range(0,len(self.utility[0,:])):
				num = math.sqrt(2*p)
				self.R[j,k] = self.R[j,k]/num
		
		print self.R 

	def U_calc(self):

		"""generates the u matrix with the help of intersection of the C and R matrix found by 
		   selecting rows and columns randomly with a certaing probability associated with the 
		   					selection of that row or column"""
		self.ucal = self.R[:,self.c]
		svd = SVD()
		svd.Ucalc(self.ucal,self.ucal.transpose())
		svd.Vcalc(self.ucal,self.ucal.transpose())
		svd.Sigma()

		sigma = svd.sigma

		for i in range(0,max(svd.rank_u,svd.rank_v)):
			if sigma[i,i]!=0:
				sigma[i,i]=(1/sigma[i,i])

		self.U = (svd.V.transpose())*(sigma)*(sigma)*(svd.U.transpose())
		
	def ErrorEstimate(self,k):

		"""determing the error in making use of CUR for estimation of the 
							ratings"""
		error = ErrorEstimation(self.C,self.U,self.R)

		"""error estimation """
		print "RMSE",
		error.RMSE(self.utility)
		print error.Rmse

		error.TopK(k,self.utility)
		print "TopK",
		print error.TopK

		print "SpearmansCorrelation",
		error.SpearmansCorrelation(self.utility)

if __name__=='__main__':
	cur = CUR("/home/nikhil/recomendorsys/utility.xlsx")
	print "Enter whether you want to sample out rows and columns with replacement or without replacement:"
	print "Enter 1 for sampling without replacement"
	print "Enter 2 for sampling with replacement"
	decision_var = input()
	cur.dataframe()
	cur.probdistribution()
	cur.C_calc(decision_var)
	cur.R_calc(decision_var)
	cur.U_calc()
	print "Enter the k value for error estimation for top k error precision"
	k = input()
	cur.ErrorEstimate(k)