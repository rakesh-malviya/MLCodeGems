import pandas as pd
import os
folder = "output2"
dfList = []
for fileName in os.listdir(folder):
  filePath = folder + "/"+fileName 
  print(filePath) 
  dfList.append(pd.read_csv(filePath))
  
folder = "output3"
for fileName in os.listdir(folder):
  filePath = folder + "/"+fileName 
  print(filePath) 
  dfList.append(pd.read_csv(filePath))

  
  
for i in range(1,len(dfList)):
  dfList[0]['click'] += dfList[i]['click']
  
dfList[0]['click'] = dfList[0]['click']/len(dfList)
dfList[0].to_csv('avgFolder_out_2_3_.csv', index=False)