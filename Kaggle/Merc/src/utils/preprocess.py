import pandas as pd
import numpy as np
"""Categorical Data Handling"""

def dfLabelEncoding(df_train,df_test):
  from sklearn.preprocessing import LabelEncoder
  le=LabelEncoder()
  for col in df_test.columns.values:
    if df_test[col].dtypes=='object':
      data = df_train[col].append(df_test[col])
#       print(data)
      le.fit(data.values)
      df_train[col]=le.transform(df_train[col])
      df_test[col]=le.transform(df_test[col])
      
  return df_train,df_test

def dfOneHotEncoding(df_train,df_test,columnList):
  from sklearn.preprocessing import OneHotEncoder
  enc=OneHotEncoder(sparse=False)
  for col in columnList:
       # creating an exhaustive list of all possible categorical values
#        print("col:",col)       
       data=df_train[[col]].append(df_test[[col]])
       enc.fit(data)
       # Fitting One Hot Encoding on train data
       temp = enc.transform(df_train[[col]])
       # Changing the encoded features into a data frame with new column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # In side by side concatenation index values should be same
       # Setting the index values similar to the df_train data frame
       temp=temp.set_index(df_train.index.values)
       # adding the new One Hot Encoded varibales to the train data frame
       df_train=pd.concat([df_train,temp],axis=1)
       # fitting One Hot Encoding on test data
       temp = enc.transform(df_test[[col]])
       # changing it into data frame and adding column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # Setting the index for proper concatenation
       temp=temp.set_index(df_test.index.values)
       # adding the new One Hot Encoded varibales to test data frame
       df_test=pd.concat([df_test,temp],axis=1)  
  
#   print(len(df_test.columns))  
  df_train.drop(columnList, axis=1, inplace=True)
  df_test.drop(columnList, axis=1, inplace=True)  
#   print(len(df_test.columns))
  
  return df_train,df_test
      

def dfMyOneHotEncoding(df_train,df_test,columnList):
  twoCharColList=[]
  
  uniqueSetTrain = set()
  for col in columnList:
#     print("\n"+col)
    tempSet = set(df_train[col])
    
    for x in tempSet:
      if(len(x)==2):
         twoCharColList.append(col)
         break
                     
    uniqueSetTrain = uniqueSetTrain | tempSet    
#     print(tempSet)    
#   print("\uniqueSetTrain: "+str(len(uniqueSetTrain))+" "+str(uniqueSetTrain))
  
  
  uniqueSetTest = set()
  for col in columnList:
#     print("\n"+col)
    tempSet = set(df_test[col])
    
    for x in tempSet:
      if(len(x)==2):
         twoCharColList.append(col)
         break
    
    uniqueSetTest = uniqueSetTest | tempSet
#     print(tempSet)    
#   print("\nuniqueSetTest: "+str(len(uniqueSetTest))+" "+str(uniqueSetTest))  
  twoCharColList = list(set(twoCharColList))
  print("twoCharColList",twoCharColList)
    
  uniqueCats = uniqueSetTest | uniqueSetTrain
  print("uniqueCats",uniqueCats)
  
  def splitFunc0(val):
    if(len(val)==2):
      return val[0]
    else:
      return '?'
    
  def splitFunc1(val):
    if(len(val)==2):
      return val[1]
    else:
      return val[0]   
    
#   df = np.vectorize(splitFunc0)(df_train[columnList[0]])
  
  newColunmList = [x for x in columnList if x not in twoCharColList]
  print(newColunmList)
  
  for col in twoCharColList:
    str0 = col+"_"+"0"
    newColunmList.append(str0)
    df_train[str0] = np.vectorize(splitFunc0)(df_train[col])
    df_test[str0] = np.vectorize(splitFunc0)(df_test[col])
    
    str1 = col+"_"+"1"
    newColunmList.append(str1)
    df_train[str1] = np.vectorize(splitFunc1)(df_train[col])
    df_test[str1] = np.vectorize(splitFunc1)(df_test[col])
    
         
  print(newColunmList)
#   print(len(df_test.columns))  
  df_train.drop(twoCharColList, axis=1, inplace=True)
  df_test.drop(twoCharColList, axis=1, inplace=True)  
  
  
  data = df_train[newColunmList].append(df_test[newColunmList])
  uniquedata = set(list(data.values.flatten()))
  print(uniquedata)
  
  dummyDf = pd.Series(list(uniquedata))
  dummyDf = pd.get_dummies(dummyDf)
  
  def npOneHot(x):
    return dummyDf[x].values
  
  
  for col in newColunmList:
    colData = df_train[col].values
#     print(colData.shape)
    newColDataList = []
    for x in colData:
      newColDataList.append(npOneHot(x))
      
    newColData = np.array(newColDataList)
#     print(newColData.shape)
    
    for colIndex in range(newColData.shape[1]):
      strCol = col+"_"+str(colIndex)
      df_train[strCol]= newColData[:,colIndex]
      
    colData = df_test[col].values
#     print(colData.shape)
    newColDataList = []
    for x in colData:
      newColDataList.append(npOneHot(x))
      
    newColData = np.array(newColDataList)
#     print(newColData.shape)
    
    for colIndex in range(newColData.shape[1]):
      strCol = col+"_"+str(colIndex)
      df_test[strCol]= newColData[:,colIndex]
      
  df_train.drop(newColunmList, axis=1, inplace=True)
  df_test.drop(newColunmList, axis=1, inplace=True)
  
                
  
#   df_train["Hello"] = df_train[df_train.columns.values[0]]
#   df_test["Hello"] = df_test[df_test.columns.values[0]]
  
  return df_train,df_test
  
  
                