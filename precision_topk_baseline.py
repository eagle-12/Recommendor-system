import pandas as pd
import xlrd
import math
import csv


class precision_topkbaseline():
	def topk_baseline(self):

		path = '/home/tex/Documents/IR/Dataset_AS3/'
		filename = 'similarity5000.csv'
		with open(path+filename, 'r') as f:
		  reader = csv.reader(f)
		  your_list = list(reader)

		sim = []
		for i in range(len(your_list)):
		    sim.append([])
		    for j in range(len(your_list[i])):
		        if your_list[i][j]=='':
		            pass
		        else:
		            sim[i].append(float(your_list[i][j]))

		filename2 = 'global_baseline5000.csv'
		with open(path+filename2, 'r') as f:
		  reader = csv.reader(f)
		  your_list1 = list(reader)

		baseline = []
		for i in range(len(your_list1)):
		    baseline.append([])
		    for j in range(len(your_list1[i])):
		        if your_list1[i][j]=='':
		            pass
		        else:
		            baseline[i].append(float(your_list1[i][j]))

		filename3 = '5000dataset.xls'

		xlsfile = xlrd.open_workbook(path+filename3, on_demand= True)
		xlsfiledata = xlsfile.sheet_by_index(0)

		data=[]
		for i in range(0,5000):
		    data.append([])
		    for j in range(1,101):
		        data[i].append(xlsfiledata.cell(i,j).value)

		error1=0
		error2=0
		for i in range(0,5000):
			print(i)
			sorted_row = []
			for j in range(0,100):
				sorted_row.append([])
				sorted_row[j].append(data[i][j])
				sorted_row[j].append(j)

			sorted_row.sort(key=lambda tup: tup[0])
			#print(sorted_row)
			top_10_index = []
			count=0
			for j in range(0,100):
				if(sorted_row[100-j-1][0]!=99):
					top_10_index.append(sorted_row[100-j-1][1])
					count=count+1
					if count>=10:
						break

			#print(top_10_index)
			for j in range(0,len(top_10_index)):
				sum1=0
				sum2=0
				for k in range(0,5000):
					if k==i:
						pass
					else:
						if i<k and sim[i][k-i]>0 and data[k][top_10_index[j]]!=99:
							sum1 = sum1 + (float(data[k][top_10_index[j]] - baseline[k][top_10_index[j]])*float(sim[i][k-i]))
							sum2 = sum2 + float(sim[i][k-i])
						elif i>k and sim[i-k][k]>0 and data[k][top_10_index[j]]!=99:
							sum1 = sum1 + (float(data[k][top_10_index[j]]-baseline[k][top_10_index[j]])*float(sim[i-k][k]))
							sum2 = sum2 + float(sim[i-k][k])
				value = sum1/sum2
				error1 = error1 + ((value-data[i][j]) * (value-data[i][j]))
				error2 = error2 + 1

		xx = math.sqrt(error1)		
		print("Precision Top 10 (Movie ratings of each user) ",xx/error2)

if __name__=='__main__':
	pt = precision_topkbaseline()
	pt.topk_baseline()
"""
Precision Top 10 (Movie ratings of each user)  0.2033023392686095
"""