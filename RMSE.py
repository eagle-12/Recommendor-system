import pandas as pd
import xlrd
import math
import csv

class RMSE_colaborativefiltering:
    """
    - This function is used for calculating root mean square error and 
      spearsman's coefficiant in obtained (recommended) values.
    """
    def RMSE(self):    
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

        filename2 = '5000dataset.xls'

        xlsfile = xlrd.open_workbook(path+filename2, on_demand= True)
        xlsfiledata = xlsfile.sheet_by_index(0)

        data=[]
        for i in range(0,5000):
            data.append([])
            for j in range(1,101):
                data[i].append(xlsfiledata.cell(i,j).value)

        error1=0
        error2=0
        for i in range(4000,5000):
            #print(i)
            for j in range(50,100):        
                if data[i][j]!=99:
                    sum1=0
                    sum2=0
                    for k in range(0,5000):
                        if k==i:
                            pass
                        else:
                            if i<k and sim[i][k-i]>0 and data[k][j]!=99:
                                sum1 = sum1 + (float(data[k][j])*float(sim[i][k-i]))
                                sum2 = sum2 + float(sim[i][k-i])
                            elif i>k and sim[i-k][k]>0 and data[k][j]!=99:
                                sum1 = sum1 + (float(data[k][j])*float(sim[i-k][k]))
                                sum2 = sum2 + float(sim[i-k][k])

                    value = sum1 / sum2
                    error1 = error1 + ((value-data[i][j]) * (value-data[i][j]))
                    error2 = error2 + 1

        xx=math.sqrt(error1)
        print('Root Mean Square Error',xx/error2)
        print('Spearman\'s Correlation',(1-((6*error1)/(error2*((error2*error2)-1)))))

if __name__ == '__main__':
    rmse = RMSE_colaborativefiltering()
    rmse.RMSE()
"""
Root Mean Square Error 0.028294626855720265
Spearman's Correlation 0.9999998352308352
"""