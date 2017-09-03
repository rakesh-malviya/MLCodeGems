import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import src.utils.loadsave as ls
from collections import Counter



# df_train = pd.read_csv('input/train.csv')
# df_test = pd.read_csv('input/test.csv')


# trainCols = list(df_train.columns)
# trainCols.remove('y')
# testCols = list(df_test.columns)
# 
# print(trainCols)
# print(testCols)
# print (trainCols == testCols)
"""Remove duplicate columns"""
def removeDuplicate(df_data):
  return  df_data.T.drop_duplicates().T
    
def removeDupTestTrain(df_test,df_train):
  df_test_column_old = set(df_test.columns.values)
  df_train_column_old = set(df_train.columns.values)    
  df_test = removeDuplicate(df_test)    
  df_train = removeDuplicate(df_train)    
  df_test_column_new = set(df_test.columns.values)
  df_train_column_new = set(df_train.columns.values)
  
  df_test_column_removed = df_test_column_old - df_test_column_new
  df_train_column_removed = df_train_column_old - df_train_column_new
#   print("df_test_column_removed",len(df_test_column_removed),df_test_column_removed)
#   print("df_train_column_removed",len(df_train_column_removed),df_train_column_removed)
  df_common_removed = df_test_column_removed & df_train_column_removed
#   print("df_common_removed",len(df_common_removed),df_common_removed)
#   print(len(df_test_column_new))
#   print(len(df_train_column_new))  
  
  return list(df_common_removed)
  
def plotY(ar):
  val = 1.0 
  plt.plot(ar, np.zeros_like(ar) + val, 'x')
  plt.show()  

def preprocessDF(trainFilename,testFilename):    
    
    df_train = pd.read_csv(trainFilename)
    df_test = pd.read_csv(testFilename)
    
    #checking outlier
    print(df_train.values.shape)
    print(df_test.values.shape)
#     plotY(df_train['y'])
    #remove outlier 
    df_train = df_train.drop(df_train[df_train['y'] >140].index)
    
#     plotY(df_train['y'])
    print(df_train.values.shape)
    print(df_test.values.shape)
    
    """Get y"""
    y_train = df_train['y'].values    
    df_train.drop("y", axis=1, inplace=True)  
    
    id_train = df_train['ID'].values    
    df_train.drop("ID", axis=1, inplace=True)
    
    id_test = df_test['ID'].values    
    df_test.drop("ID", axis=1, inplace=True)
    
    
    """Handle Catgorical Data"""
    from src.utils import preprocess
    
    catColList = []
    for col in df_test.columns.values:
      if df_test[col].dtypes=='object':
        catColList.append(col)
    
    
    print(catColList)   
    
    print(len(df_test.columns))
    print(len(df_train.columns))
#     df_train, df_test = preprocess.dfLabelEncoding(df_train, df_test)
#     df_train, df_test = preprocess.dfOneHotEncoding(df_train, df_test,catColList)
    
    df_train, df_test = preprocess.dfMyOneHotEncoding(df_train, df_test,catColList)
    print(len(df_test.columns))
    print(len(df_train.columns))
    
        
    duplicateColumns = removeDupTestTrain(df_test, df_train)
    df_train.drop(duplicateColumns, axis=1, inplace=True)
    df_test.drop(duplicateColumns, axis=1, inplace=True)   
    
    print(len(df_test.columns))
    print(len(df_train.columns))
    
    X_train = df_train.values
    X_test =  df_test.values
    
    
    
    return X_train,y_train,id_train,X_test,id_test

trainFile = 'input/train.csv'
testFile = 'input/test.csv'

X_train,y_train,id_train,X_test,id_test = preprocessDF(trainFile,testFile)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print("Xtrain max:",np.amax(X_train,axis=(0,1)))
print("X_test max:",np.amax(X_test,axis=(0,1)))
print("Xtrain min:",np.amin(X_train,axis=(0,1)))
print("X_test min:",np.amin(X_test,axis=(0,1)))   
# exit()    
    
import xgbTrain,xgbTrainGrid,KerasMain
# xgbTrain.avging(id_test)
# xgbTrain.xgboost(id_test, y_train, X_train, X_test)
xgbTrain.xgboostScikit(id_test, y_train, X_train, X_test)

# KerasMain.KerasTrain(id_test, y_train, X_train, X_test)

# xgbTrainGrid.xgboostGridRegress(id_test, y_train, X_train, X_test)

# xgbTrain.xgboostGridSearch(id_test, y_train, X_train, X_test)

# xgbTrainGrid.xgboostRandomRegress(id_test, y_train, X_train, X_test)
