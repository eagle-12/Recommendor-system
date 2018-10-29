import numpy
import pandas as pd
import xlwt
import xlrd
import math
import csv

class similarity_calculation:
	def similarity_matrix(self):
		"""
		-This Function is Used to Calculate similarity matrix of 5000*100 
		 dataset elements which has user*movie_ratings a its dimentions.
		-Opening and closing of given xls file is done using xlrd and xlwt
		-Row Average(Average Rating Given By a user) is simultaniously 
		 calculated while reading the data
		"""
		
		path = '/home/tex/Documents/IR/Dataset_AS3/'
		filename = 'finaldata.xls'
		#data = pd.ExcelFile(path+filename)

		xlsfile = xlrd.open_workbook(path+filename, on_demand= True)
		xlsfiledata = xlsfile.sheet_by_index(0)

		#print(xlsfiledata.cell(0,0).value)

		rowavg =[]
		for i in range(0,5000):
		    sum=0
		    #print(i)
		    for j in range(1,101):
		        value = xlsfiledata.cell(i,j).value
		        if value>=-10 and value<=10:
		            sum=sum+value
		    rowavg.append(sum/xlsfiledata.cell(i,0).value)

		"""
		-Similarity matrix is a sort of adjecency matrix which contains
		 similarity of a user i with user j where i>j for all n users
		"""
		similarity = []
		for i in range(0,5000):
		    similarity.append([])
		    print(i)
		    similarity[i].append(1)
		    for j in range(i+1,5000):
		        sum1=0
		        sum2=0
		        sum3=0
		        for k in range(1,101):
		            val1 = xlsfiledata.cell(i,k).value
		            val2 = xlsfiledata.cell(j,k).value
		            if val1!=99 and val2!=99:
		                sum1=sum1+((val1-rowavg[i])*(val2-rowavg[j]))
		                sum2=sum2+((val1-rowavg[i])*(val1-rowavg[i]))
		                sum3=sum3+((val1-rowavg[j])*(val1-rowavg[j]))

		        sim= sum1/((math.sqrt(sum2))*(math.sqrt(sum3)))
		        similarity[i].append(sim)

		#print(len(similarity[18]))
		my_df = pd.DataFrame(similarity)
		my_df.to_csv(path+'similarity5000.csv',index=False,header=False)

if __name__=='__main__':
	sc = similarity_calculation()
	sc.similarity_matrix()
