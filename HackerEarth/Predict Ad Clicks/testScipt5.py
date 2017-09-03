import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn import metrics
import loadsave as ls
from collections import Counter

def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)
def printConfMat(y_true, y_pred):
  confMat=(metrics.confusion_matrix(y_true, y_pred))
  print(" ")
  print(confMat)
  print(0,confMat[0][0]/(confMat[0][0]+confMat[0][1]))
  print(1,confMat[1][1]/(confMat[1][1]+confMat[1][0]))

def get_class_weights(y, smooth_factor=0):
    """
     values around 0.1 (smooth factor) are a good default for very imbalanced classes.
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


isPrepro = False
if isPrepro:
  print("Reading data.....")
  train = pd.read_csv("input/train.csv")
  test = pd.read_csv("input/test.csv")
  
  # check missing values per column
  train.isnull().sum(axis=0)/train.shape[0]
  
  train['siteid'].fillna(-999, inplace=True)
  test['siteid'].fillna(-999, inplace=True)
  
  train['browserid'].fillna("None", inplace=True)
  test['browserid'].fillna("None", inplace=True)
  
  train['devid'].fillna("None", inplace=True)
  test['devid'].fillna("None", inplace=True)
  
  # set datatime
  train['datetime'] = pd.to_datetime(train['datetime'])
  test['datetime'] = pd.to_datetime(test['datetime'])
  
  # create datetime variable
  train['tweekday'] = train['datetime'].dt.weekday
  train['thour'] = train['datetime'].dt.hour
  train['tminute'] = train['datetime'].dt.minute
  
  test['tweekday'] = test['datetime'].dt.weekday
  test['thour'] = test['datetime'].dt.hour
  test['tminute'] = test['datetime'].dt.minute
  
  print("Processing data.....")
  cols = ['siteid','offerid','category','merchant']
  
  for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')
    
  cat_cols = cols + ['countrycode','browserid','devid']
  
  for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values) + list(test[col].values))
    train[col] = lbl.transform(list(train[col].values))
    test[col] = lbl.transform(list(test[col].values))
    
  cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))
  # X_train, X_test, y_train, y_test = train_test_split(train[cols_to_use], train['click'], test_size = 0.5)
  X_train = train[cols_to_use].values
  y_train = train['click'].values

X_train = ls.load_obj("X_train")
y_train = ls.load_obj("y_train")
testID = ls.load_obj("testID")
test = ls.load_obj("test")
click_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
print(np.unique(y_train),click_weight)
print("Training data.....")

from sklearn.model_selection import KFold
import numpy as np
foldCount = 3 
kf = KFold(n_splits=foldCount, random_state=52, shuffle=True)
fPreds = np.zeros(len(test))
aucScore = 0
 
for train_index, test_index in kf.split(X_train):
    
  X_tr, X_te = X_train[train_index], X_train[test_index]
  y_tr, y_te = y_train[train_index], y_train[test_index]

  dtrain = lgb.Dataset(X_tr, y_tr)
  dval = lgb.Dataset(X_te, y_te)
  click_weight = get_class_weights(y_tr)
  params = {
      
      'num_leaves' : 256,
      'learning_rate':0.02,
      'metric':'auc',
      'objective':'binary',
      'early_stopping_round': 40,
      'max_depth':10,
      'bagging_fraction':0.5,
      'feature_fraction':0.6,
      'bagging_seed':2017,
      'feature_fraction_seed':2017,
      'verbose' : 1,
      'scale_pos_weight': click_weight[1]
  }
  
  clf = lgb.train(params, dtrain,num_boost_round=500,valid_sets=dval,verbose_eval=20)
  yPred=clf.predict(X_te)
  tempScore = auc(y_te, yPred)
  yPred[yPred<0.5]=0
  yPred[yPred>=0.5]=1
  valPredClass =yPred
  printConfMat(y_te,valPredClass)
  aucScore += tempScore
  print("tempScore",tempScore)
  
  preds = clf.predict(test)
  
  
  fPreds += preds

print("aucScore",aucScore)
aucScore = aucScore/foldCount
fPreds = fPreds/foldCount
sub = pd.DataFrame({'ID':testID, 'click':fPreds})
sub.to_csv('lgb_pyst_5_%s.csv' %(str(aucScore)), index=False)


    
    
    
