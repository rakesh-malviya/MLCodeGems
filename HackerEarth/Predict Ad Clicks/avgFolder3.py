import pandas as pd
import os
folder = "output2"
fileList = [("Keras_2_0.96762186875.csv",8),
            ("Keras_3_0.95938402746.csv",8),
            ("KerasLSTM_1_0.988391978129.csv",4),
            ("KerasLSTM_1_0.988391978129.csv",4),
            ("lgb_pyst_6_0.972273561091.csv",8),
            ("lgb_pyst_7_0.972286703622.csv",8)           
            ]
dfList = []
for fileName,w in fileList:
  filePath = folder + "/"+fileName 
  print(filePath) 
  dfList.append([pd.read_csv(filePath),w])
  
sumW = 0
for i in range(0,len(dfList)):
  sumW += dfList[i][1]
  if i==0:
    dfList[0][0]['click'] = dfList[i][0]['click']*dfList[i][1]
  else:
    dfList[0][0]['click'] += dfList[i][0]['click']*dfList[i][1]
    
  
dfList[0][0]['click'] = dfList[0][0]['click']/sumW
dfList[0][0].to_csv('avgFolder_out2_l23_k67_lstm1_8_8_4.csv', index=False)