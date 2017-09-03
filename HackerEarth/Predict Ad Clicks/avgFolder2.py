import pandas as pd
import os
folder = "output2"
fileList = []
dfList = []
for fileName in os.listdir(folder):
  filePath = folder + "/"+fileName 
  print(filePath) 
  fileList.append(fileName)
  dfList.append(pd.read_csv(filePath))
  
folder = "output3"
for fileName in os.listdir(folder):
  filePath = folder + "/"+fileName 
  print(filePath)
  fileList.append(fileName) 
  dfList.append(pd.read_csv(filePath))

sum = 0  
  
for i in range(1,len(dfList)):
  fileName = fileList[i]
  mu = None
  if "avg" in fileName:
    mu = 6
  elif "Keras" in fileName:
    mu = 6
  elif "LSTM" in fileName:
    mu = 1
  elif "lgb" in fileName:
    mu = 2
  
  sum+=mu
  dfList[0]['click'] += (dfList[i]['click']*mu)
  
dfList[0]['click'] = dfList[0]['click']/sum
dfList[0].to_csv('avgFolder_Ratio_out_2_3_.csv', index=False)