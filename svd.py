import pandas as pd
import numpy as np
import numpy.linalg as linalg
import math

class SVD:
	def __init__(self, file=None):
		self.filename = file
	
	
	def dataframe(self):
		"""generates a utility matrix in the form of numpy array"""
		utilitymatrix = pd.read_excel(self.filename,headers = None)
		utility = utilitymatrix.values
		utility_transpose = utility.transpose()
		return (utility, utility_transpose)
	


	def Ucalc(self, utility, utility_transpose):
		"""finds the U matrix for the given matrix by first calculating
		     A*A(TRANSPOSE) where A is the original utility matrix 
								available"""

		U_calc = np.dot(utility,utility_transpose)

		"""determining the eigen values and eigen vectors corresponding 
						to the matrix obtained"""
		self.rank_u = linalg.matrix_rank(U_calc)
		self.eigen_values_u,self.eigenvectors_u = linalg.eigh(U_calc)
		self.num_u = len(self.eigen_values_u)


		"""sorting the eigen values and corresponding eigenvectors"""
		idx = self.eigen_values_u.argsort()[::-1]   
		self.eigen_values_u = self.eigen_values_u[idx]
		self.eigenvectors_u = self.eigenvectors_u[:,idx]

		"""selection of the largest rank_u eigen values of the matrix"""
		self.eig_vals_sorted_u = []

		for i in range(0,self.rank_u):
			self.eig_vals_sorted_u.append(self.eigen_values_u[i])

		self.eigenvectors_u = self.eigenvectors_u[:,0:self.rank_u]

		
		"""U matrix which is obtained by placing the eigen vectors in 
								column"""
		self.U = self.eigenvectors_u


	


	def Vcalc(self, utility, utility_transpose):

		"""finds the V matrix for the given matrix by first calculating
		     A(TRANSPOSE)*A where A is the original utility matrix 
								available"""
		V_calc = np.dot(utility_transpose,utility)

		"""determining the eigen values and eigen vectors corresponding 
						to the matrix obtained"""
		self.rank_v = linalg.matrix_rank(V_calc)
		self.eigen_values_v,self.eigenvectors_v = linalg.eigh(V_calc)
		self.num_v = len(self.eigen_values_v)

		"""sorting the eigen values and corresponding eigenvectors"""
		idx = self.eigen_values_v.argsort()[::-1]   
		self.eigen_values_v = self.eigen_values_v[idx]
		self.eigenvectors_v = self.eigenvectors_v[:,idx]

		"""selecting the largest rank_u eigen values of the matrix"""
		self.eig_vals_sorted_v = []

		print self.eigen_values_v

		for i in range(0,self.rank_v):
			self.eig_vals_sorted_v.append(self.eigen_values_v[i])

		print self.eig_vals_sorted_v
		self.eigenvectors_v = self.eigenvectors_v[:,0:self.rank_v]


		"""V matrix which is obtained by placing the eigen vectors in 
								rows"""
		self.V = self.eigenvectors_v.transpose()

	



	def Sigma(self):
		"""sigma calculation for the given matrix by taking the square 
					root of the eigen values"""
		if self.rank_u < self.rank_v:
			self.sigma = np.diag(np.sqrt(self.eig_vals_sorted_u))
		else:
			self.sigma = np.diag(np.sqrt(self.eig_vals_sorted_v))

	


	def DimentionReduction(self):

		"""It is better to retain only those singular values that 
			make up 90% of the energy of Sigma matrix i.e. the sum 
			of the squares of the retained values should be 90% of 
			the sum of the squares of all the singular values"""

		print "inside"
		sum_squares_singular_vals = 0
		for i in range(0,min(self.rank_u,self.rank_v)):
			sum_squares_singular_vals = sum_squares_singular_vals + self.sigma[i,i]*self.sigma[i,i]

		idx = min(self.rank_u,self.rank_v)
		least_energy_val = self.sigma[idx-1,idx-1]
		
		if ((least_energy_val*least_energy_val)/(sum_squares_singular_vals))<0.1:

			"""changing the U matrix i.e. deleting the last column of 
				             the U matrix"""
			self.U = self.U[:,0:(self.rank_u-1)]
			print "#"
			print self.U

			"""changing the V matrix i.e. deleting the last column of 
				             the U matrix"""
			self.V = self.V[0:(self.rank_v-1),:]

			"""changing the sigma matrix i.e. deleting the last row as
			               well as last column"""
			self.sigma = self.sigma[0:(idx-1),0:(idx-1)]

		else:
			print "reduction is not to be done"




class ErrorEstimation:
	def __init__(self, U, Sigma, V):

		"""obtain the approximated matrix by multiplication of U,Sigma 
								and V"""
		self.approx = np.dot(np.dot(U,Sigma),V)
		print "approx"
		print self.approx
	def RMSE(self, original):

		"""computing the root mean squared errors by taking the square root
		of the sum of the squares of the difference of the known ratings"""

		self.Rmse = 0
		num_vals = 0

		for i in range(0,len(original)):
			for j in range(0,len(original[0])):
				if original[i,j]!=0:
					num_vals += 1
					self.Rmse += (self.approx[i,j]-original[i,j])*(self.approx[i,j]-original[i,j])

		self.Rmse = math.sqrt(self.Rmse)/num_vals

	def TopK(self, k, original):

		"""Selecting the top k elements of the matrix and estimating rmse values 
				for the selected K values. First obtaining all the values in a 
								list"""
		topk = []
		for i in range(0,len(original)):
			for j in range(0,len(original[i])):
				topk.append((original[i,j],i,j))

		""""""		
		topk.sort()


		self.topk_Rmse = 0

		for i in range(0,k):
			rowidx = topk[len(topk)-1-i][1]
			colidx = topk[len(topk)-1-i][2]
			print rowidx
			print colidx
			self.topk_Rmse += (topk[len(topk)-1-i][0]-(self.approx[rowidx,colidx]))*(topk[len(topk)-1-i][0]-(self.approx[rowidx,colidx]))

		self.topk_Rmse = math.sqrt(self.topk_Rmse)/k

	def SpearmansCorrelation(self, original):

		"""determining the spearman's correlation by making use of 
		             the correlation coefficient"""

		diff = 0
		num_val = 0

		for i in range(0,len(original)):
			for j in range(0,len(original[0])):
				num_val +=1
				if original[i,j]!=0:
					diff += (self.approx[i,j]-original[i,j])*(self.approx[i,j]-original[i,j])					
					
		print 6*diff
		print num_val
		self.correlationcoef = 1 - ((6*diff)/(num_val*(num_val**2-1)))

		print self.correlationcoef


if __name__=='__main__':
	svd = SVD("/home/nikhil/recomendorsys/utility.xlsx")
	(utility, utility_transpose) = svd.dataframe()
	print "utility matrix :"
	print utility
	print len(utility[0])
	svd.Ucalc(utility, utility_transpose)
	svd.Vcalc(utility, utility_transpose)		
	print svd.U
	print svd.V
	svd.Sigma()
	print svd.sigma
	print "Is Dimention Reduction Required or not:"
	print "Enter 1 if required"

	decision_var = input()
	if decision_var==1:
		svd.DimentionReduction()

	error = ErrorEstimation(svd.U,svd.sigma,svd.V)
	error.RMSE(utility)
	print "rmse"
	print error.Rmse

	error.TopK(4, utility)
	print "TopK"
	print error.topk_Rmse

	error.SpearmansCorrelation(utility)