import src.utils.loadsave as ls
import numpy as np
import pandas as pd
from collections import Counter

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
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

def xg_eval_r2(yhat, dtrain):
    y = dtrain.get_label()
    return 'r2_score', r2_score(y, yhat)
  
def xgboostGridRegress(ids,train_y,train_x,test_x):    
#     ids,train_y,train_x,test_x =  load_obj("saveAliData")
    from sklearn.cross_validation import KFold
    seed = 2016
    cv = KFold(len(train_y),n_folds=10, shuffle=True, random_state=seed)
    
    impIndexList = checkImportancesUnion()
    train_x = train_x[:, impIndexList]
    test_x = test_x[:, impIndexList]

    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    
    """
            params = {
            'seed': 0,
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.001,
            'objective': 'reg:linear',
            'max_depth': 12,
            'min_child_weight': 100,
            'booster': 'gbtree'}
    """
   

    params_grid = {
      'max_depth': [5,9,12, 15, 20],
      'colsample_bytree': [0.3,0.5,0.7,0.9],
      'subsample': [0.3,0.5,0.7,0.9],
      'n_estimators': [25, 50,100,200],
      'learning_rate': [0.0005,0.001,0.003,0.005,0.009],
      'min_child_weight':[30,50,90,120]
    }


    
    params_fixed = {
                    'booster': 'gbtree',
                    'objective': 'reg:linear',
                    'silent': 1
    }
    
    rs_grid = GridSearchCV(
      estimator=XGBRegressor(params_fixed, seed=seed),
      param_grid=params_grid,
      cv=cv,
      scoring='neg_mean_absolute_error'
    )
    
    
    searchOn =False
    if(searchOn):
      print("Starting search")    
      rs_grid.fit(train_x, train_y)     
      best_estimator = rs_grid.best_estimator_
      best_params  = rs_grid.best_params_
      print(best_params)
      best_score = rs_grid.best_score_
      print(best_score)
      ls.save_obj(rs_grid, "rs_grid")
      ls.save_obj(best_estimator, "best_estimator")
      ls.save_obj(best_params, "best_params")
      ls.save_obj(best_score, "best_score")
    else:
      best_estimator = ls.load_obj("best_estimator")
      score = ls.load_obj("best_score")
#       best_estimator = (XGBRegressor)best_estimator
      mpred = best_estimator.predict(test_x)      
      print(type(best_estimator))
      
      print("Writing results")
      result = pd.DataFrame(mpred, columns=['y'])
      result["ID"] = ids
      result = result.set_index("ID")
      
      
      sub_file = 'output/XGB_1_submission_gridsearch_R2' + str(score) + '.csv'
      print("Writing submission: %s" % sub_file)
      result.to_csv(sub_file, index=True, index_label='id')
    
    print("done")
  
  

def xgboostRandomRegress(ids,train_y,train_x,test_x):    
#     ids,train_y,train_x,test_x =  load_obj("saveAliData")
    from sklearn.cross_validation import KFold
    seed = 2016
    cv = KFold(len(train_y),n_folds=10, shuffle=True, random_state=seed)
    
    impIndexList = checkImportancesUnion()
    train_x = train_x[:, impIndexList]
    test_x = test_x[:, impIndexList]

    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    
    """
            params = {
            'seed': 0,
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.001,
            'objective': 'reg:linear',
            'max_depth': 12,
            'min_child_weight': 100,
            'booster': 'gbtree'}
    """
    params_dist_grid = {
      'max_depth': randint(6, 16),
      'gamma': [0, 0.5, 1],
      'n_estimators': randint(1, 1001), # uniform discrete random distribution
      'learning_rate': uniform(), # gaussian distribution
      'subsample': uniform(), # gaussian distribution
      'colsample_bytree': uniform() # gaussian distribution
    }
    
    
    params_fixed = {
                    'booster': 'gbtree',
                    'seed': seed,
                    'objective': 'reg:linear',
                    'silent': 1
    }
    
    rs_grid = RandomizedSearchCV(
        estimator=XGBRegressor(params_fixed, seed=seed),
        param_distributions=params_dist_grid,
        n_iter=10,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=seed
    )
    
    
    searchOn =False
    if(searchOn):
      print("Starting search")    
      rs_grid.fit(train_x, train_y)     
      best_estimator = rs_grid.best_estimator_
      best_params  = rs_grid.best_params_
      print(best_params)
      best_score = rs_grid.best_score_
      print(best_score)
#       ls.save_obj(rs_grid, "rs_grid")
#       ls.save_obj(best_estimator, "best_estimator")
#       ls.save_obj(best_params, "best_params")
#       ls.save_obj(best_score, "best_score")
    else:
      best_estimator = ls.load_obj("best_estimator")
      score = ls.load_obj("best_score")
      best_params = ls.load_obj("best_params")
      print(best_estimator)
#       best_estimator = (XGBRegressor)best_estimator
      mpred = best_estimator.predict(test_x)      
      print(type(best_estimator))
      
      print("Writing results")
      result = pd.DataFrame(mpred, columns=['y'])
      result["ID"] = ids
      result = result.set_index("ID")
      
      
      sub_file = 'output/XGB_1_submission_randomgridsearch_R2' + str(score) + '.csv'
      print("Writing submission: %s" % sub_file)
      result.to_csv(sub_file, index=True, index_label='id')
    
    print("done")

    
    