import src.utils.loadsave as ls
import numpy as np
import pandas as pd
from collections import Counter
from copy import deepcopy
from pprint import pprint
import os


def checkImportancesUnion():
  importancesList = ls.load_obj("importancesList")
  importancesKeySetList = []

  for importances in importancesList:
    importancesKeySetList.append(set(importances.keys()))
  
  commonSet =  importancesKeySetList[0]

  for i in range(1,len(importancesKeySetList)):
    commonSet = commonSet | importancesKeySetList[i]
  
  commonSet = [int(x.replace('f','')) for x in commonSet]
  print(len(commonSet),commonSet)
  return commonSet

def checkImportancesIntersect():
  importancesList = ls.load_obj("importancesList")
  importancesKeySetList = []

  for importances in importancesList:
    importancesKeySetList.append(set(importances.keys()))
  
  commonSet =  importancesKeySetList[0]

  for i in range(1,len(importancesKeySetList)):
    commonSet = commonSet & importancesKeySetList[i]  

  print(len(commonSet),commonSet)
  temp = [int(x.replace('f','')) for x in commonSet]
  temp.sort()
  fScoreSumDict = {}
  cutoff = -1
  
  for x in commonSet:
    newKey = int(x.replace('f',''))    
    for importances in importancesList:
      if importances[x]<cutoff:
        continue
      try:
        fScoreSumDict[newKey] += importances[x]
      except:
        fScoreSumDict[newKey] = importances[x]        
  
          
#   drawBars(Counter(fScoreSumDict.values()), "blue")
  
  print(len(fScoreSumDict),fScoreSumDict.values())
#   plt.hist(fScoreSumDict.values(),[x for x in xrange(0,7000,10)])
#   plt.show()      
          
  indexList = fScoreSumDict.keys()
  valueList = fScoreSumDict.values()
  print("Max",max(valueList))
  print("Min",min(valueList))
  print("Avg",sum(valueList)/float(len(valueList)))
  
#   indexList = [int(x.replace('f','')) for x in commonSet]
  print(indexList)
  return(indexList)

# checkImportancesIntersect()
# checkImportancesUnion()
# exit()

from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold

def xg_eval_r2(yhat, dtrain):
    y = dtrain.get_label()
    return 'r2_score', r2_score(y, yhat)
  
def recPermList(lenList,index=0):
  if index >= len(lenList):
    return []
  
  permList = []
    
  for i in range(lenList[index]):
    tempList = recPermList(lenList, index=index+1)
    if len(tempList)==0:
      permList.append([i])
    else:
      for j in range(len(tempList)):
        tempList[j] = [i] + tempList[j]
        
      permList += tempList
  
  return permList

# pprint(recPermList([4,4,5]))
  
  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,LearningRateScheduler

def getLr(curEpoch):
  if curEpoch < 100:
    return 0.001
  elif curEpoch < 200:
    return 0.0005
  elif curEpoch < 300:
    return 0.0001
  elif curEpoch < 450:
    return 0.00005
  else:
    return 0.00003


def nn_model(xtrain):
    model = Sequential()    
    model.add(Dense(500, input_dim = xtrain.shape[1], kernel_initializer='normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
#     model.add(Dense(600,  kernel_initializer='normal'))
#     model.add(PReLU())
#     model.add(BatchNormalization())    
#     model.add(Dropout(0.2))
#     
#     model.add(Dense(300,  kernel_initializer='normal'))
#     model.add(PReLU())
#     model.add(BatchNormalization())    
#     model.add(Dropout(0.2))
    
    model.add(Dense(1,  kernel_initializer='normal'))
    opt = Adam(lr=0.0002)
    model.compile(loss = 'mae', optimizer = opt)
    return(model)
  


def KerasTrain(ids,train_y,train_x,test_x,dirName=None,userParams=None):
    import xgboost as xgb
#     ids,train_y,train_x,test_x =  load_obj("saveAliData")
    if userParams:
      ls.save_obj(userParams,dirName+"/"+"userParams")
    impIndexList = checkImportancesUnion()
    train_x = train_x[:, impIndexList]
    test_x = test_x[:, impIndexList]

    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)

    
    n_folds = 5
    cv_sum = 0
    early_stopping = 100
    fpred = []
    xgb_rounds = []
    
    pred = 0
    importancesList = []
    kf = KFold(train_x.shape[0], n_folds=n_folds)
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        X_train, X_val = train_x[train_index], train_x[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]
        
        model = nn_model(train_x)
        model.fit(X_train,y_train,batch_size=128,nb_epoch = 500,callbacks = [LearningRateScheduler(getLr)])        
        scores_val = model.predict(X_val,batch_size=1024) 
        cv_score = r2_score(y_val,scores_val)
        print('R2 score: %.6f' % cv_score)
        y_pred = model.predict(test_x,batch_size=1024)

        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred
        cv_sum = cv_sum + cv_score

    mpred = pred / n_folds
    score = cv_sum / n_folds
    print('Average R2: %.6f' % score)

    print("Writing results")
    result = pd.DataFrame(mpred, columns=['y'])
    result["ID"] = ids
    result = result.set_index("ID")
    print("%d-fold average prediction:" % n_folds)
    
    score = str(round((cv_sum / n_folds), 6))
    
    
    if dirName:
      sub_file = 'obj/'+dirName+'/XGB_fairobj_R2' + str(score) + '.csv'
    else:    
      sub_file = 'output/Keras_2_fairobj_R2' + str(score) + '.csv'
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id')    
#     print("importancesList",importancesList)

    
    
    
    
    
    