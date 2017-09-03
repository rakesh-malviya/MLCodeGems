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
  
def avging(ids):
  scoreList = ls.load_obj("scoreList")
  print(len(scoreList),scoreList)
  avg = sum(scoreList)/float(len(scoreList))
  print(avg)
  yPredList = ls.load_obj("yPredList")
  sumY = np.zeros(yPredList[0].shape)
  print(sumY.shape)
  count=0   
  for i in range(len(scoreList)):
    if scoreList[i] > 0.65:#avg:
      count += 1
      sumY = sumY + yPredList[i]
      
  print(count)
  mpred = sumY/float(count)
  print("Writing results")
  result = pd.DataFrame(mpred, columns=['y'])
  result["ID"] = ids
  result = result.set_index("ID")
  sub_file = 'output/XGB_16_submission_5fold-average-xgb_fairobj_R2' +'.csv'
  print("Writing submission: %s" % sub_file)
  result.to_csv(sub_file, index=True, index_label='ID')
    
      
  
    
  


def xgboost(ids,train_y,train_x,test_x,dirName=None,userParams=None):
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
    
    n_folds = 10
    cv_sum = 0
    early_stopping = 100
    fpred = []
    xgb_rounds = []
    scoreList = []
    yPredList = []

    d_train_full = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x)
    pred = 0
    importancesList = []
    kf = KFold(train_x.shape[0], n_folds=n_folds)
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        X_train, X_val = train_x[train_index], train_x[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]

        rand_state = 2016
        if userParams==None:
          params = {
              'seed': rand_state,
              'colsample_bytree': 0.5,
              'silent': 1,
              'subsample': 0.5,
              'learning_rate': 0.001,
              'objective': 'reg:linear',
              'max_depth': 18,
              'min_child_weight': 50,
              'booster': 'gbtree'}
        else:
          params = userParams

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        clf = xgb.train(params,
                        d_train,
                        100000,
                        watchlist,
                        early_stopping_rounds=50)
        
        importances = clf.get_fscore()
        importancesList.append(importances)

        xgb_rounds.append(clf.best_iteration)
        scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)        
        cv_score = r2_score(y_val,scores_val)
        scoreList.append(cv_score)
        print('eval-MAE: %.6f' % cv_score)
        y_pred = clf.predict(d_test, ntree_limit=clf.best_ntree_limit)
        yPredList.append(y_pred)

        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred
        cv_sum = cv_sum + cv_score

    
    ls.save_obj(scoreList,"scoreList")
    ls.save_obj(yPredList,"yPredList")
    mpred = pred / n_folds
    score = cv_sum / n_folds
    print('Average eval-MAE: %.6f' % score)
    n_rounds = int(np.mean(xgb_rounds))

    print("Writing results")
    result = pd.DataFrame(mpred, columns=['y'])
    result["ID"] = ids
    result = result.set_index("ID")
    print("%d-fold average prediction:" % n_folds)    
    score = str(round((cv_sum / n_folds), 6))
    
    
    if dirName:
      sub_file = 'obj/'+dirName+'/XGB_fairobj_R2' + str(score) + '.csv'
    else:    
      sub_file = 'output/XGB_21_submission_R2_' + str(score) + '.csv'
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='ID')    
#     print("importancesList",importancesList)
#     ls.save_obj(importancesList, "importancesList")




def xgboostGridSearch(ids,train_y,train_x,test_x):
  prefix = "search"
  rand_state = 2016
  params = {
          'seed': rand_state,
          'colsample_bytree': [0.5,0.7,0.9],
          'silent': 1,
          'subsample': [0.5,0.7,0.9],
          'learning_rate': [0.0005,0.001,0.005],
          'objective': 'reg:linear',
          'max_depth': [9, 12, 15],
          'min_child_weight': 25,   #minimum was better
          'booster': 'gbtree'
  }
  
  
  
  searchParamsList = []
  searchParamsIndexList = []
  searchParamsLenList = []
  
  for k,v in params.items():
    if type(v)==list:
      searchParamsList.append(k)
      searchParamsIndexList.append(0)
      searchParamsLenList.append(len(v))
  
  print(searchParamsList)
  permList = recPermList(searchParamsLenList)
  
  for elemList in permList:
    userParamDict = deepcopy(params)
    dirName = prefix+"/"
    
    for paramIndex,paramListIndex in enumerate(elemList):
      curParam = searchParamsList[paramIndex]
      curParamVal = params[curParam][paramListIndex]
      dirName += curParam[:3]+"_"+str(paramListIndex)+"_"      
      userParamDict[curParam] = curParamVal
      
    
    print(dirName)
    try:
      os.mkdir("obj/"+dirName)
      xgboost(ids, train_y, train_x, test_x, dirName, userParamDict)
    except Exception as e:
      print(str(e))
      pass    
    
    
    
    
    