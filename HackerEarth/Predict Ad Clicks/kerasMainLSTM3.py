import pandas as pd
import numpy as np
# import lightgbm as lgb
import loadsave as ls
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics
from keras import backend as K
from sklearn import preprocessing
from sklearn.utils import class_weight
from collections import Counter

def printConfMat(y_true, y_pred):
  confMat=(metrics.confusion_matrix(y_true, y_pred))
  print(" ")
  print(confMat)
  print(0,confMat[0][0]/(confMat[0][0]+confMat[0][1]))
  print(1,confMat[1][1]/(confMat[1][1]+confMat[1][0]))

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))

def setDayTime(row):
  row['isWeekend'] = np.where(((row['tweekday'] == 5) | (row['tweekday'] == 6)),1,0)
  row['isLateNight'] = np.where(((row['thour'] <= 7) | (row['thour'] >= 22)),1,0)
  row['isNight'] = np.where(((row['thour'] <= 3) | (row['thour'] >= 19)),1,0)
  row['isEarlyMorn'] = np.where(((row['thour'] >= 7) & (row['thour'] <= 12)),1,0)
  row['isDay'] = np.where(((row['thour'] >= 10) & (row['thour'] <= 17)),1,0)
  row['isNoon'] = np.where(((row['thour'] >= 15) & (row['thour'] <= 21)),1,0)

def isWeekend(row):
  if row['tweekday'] == 5 or row['tweekday'] == 6:
    return 1
  else:
    return 0

def isLateNight(row):
  if row['thour'] <= 7 or row['thour'] >= 22:
    return 1
  else:
    return 0
  
def isNight(row):
  if row['thour'] <= 3 or row['thour'] >= 19:
    return 1
  else:
    return 0
  
def isEarlyMorn(row):
  if row['thour'] >= 7 and row['thour'] <= 12:
    return 1
  else:
    return 0    
 
def isDay(row):
  if row['thour'] >= 10 and row['thour'] <= 17:
    return 1
  else:
    return 0    

def isNoon(row):
  if row['thour'] >= 15 and row['thour'] <= 21:
    return 1
  else:
    return 0
isPreprocess= False      
if (isPreprocess):
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
  
  print("setting day hour .....")
  #isNoon, isDay,isEarlyMorn,isNight,isLateNight
  setDayTime(train)
  
  test['tweekday'] = test['datetime'].dt.weekday
  test['thour'] = test['datetime'].dt.hour
  test['tminute'] = test['datetime'].dt.minute
  
  setDayTime(test)
  
  print("Processing data.....")
  cols = ['siteid','offerid','category','merchant']
  
  for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')
    
  cat_cols = cols + ['countrycode','browserid','devid']
  print("LabelEncoding data.....")
  for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values) + list(test[col].values))
    train[col] = lbl.transform(list(train[col].values))
    test[col] = lbl.transform(list(test[col].values))    
  
#   print("standardscaling data.....")  
#   for col in cat_cols:
#     data=train[[col]].append(test[[col]])
#     lbl = StandardScaler() 
#     lbl.fit(data)
#     train[col] = lbl.transform(train[[col]])
#     test[col] = lbl.transform(test[[col]])
    
    
#   cols_to_use = list(set(train.columns) - set(['ID','datetime','click','tminute','thour','tweekday']))
  cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))
  # X_train, X_test, y_train, y_test = train_test_split(train[cols_to_use], train['click'], test_size = 0.5)
  print(train.columns)
  X_train = train[cols_to_use].values
  y_train = train['click'].values
  # X_train = X_train[:100]
  # y_train = y_train[:100]
  testID = test['ID']
  test = test[cols_to_use].values
  print(X_train.shape,y_train.shape)
  print("Saving data.....")
  ls.save_obj(X_train,"X_train_lstm")
  ls.save_obj(y_train,"y_train_lstm")
  ls.save_obj(testID,"testID_lstm")
  ls.save_obj(test,"test_lstm")  
  exit()
print("Training data.....")
X_train = ls.load_obj("X_train_lstm")
y_train = ls.load_obj("y_train_lstm")
testID = ls.load_obj("testID_lstm")
test = ls.load_obj("test_lstm")
testUnique = np.unique(test)
trainUnique = np.unique(X_train)
dataUnique = np.unique(np.concatenate([trainUnique,testUnique]))
print("X_train",X_train.shape)
print("testUnique",testUnique.shape)
print("trainUnique",trainUnique.shape)
print("dataUnique",dataUnique.shape)
max_features = dataUnique.shape[0]
click_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

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


def KerasModel():
  from keras.preprocessing import sequence
  from keras.models import Sequential
  from keras.layers import Dense, Embedding,Activation
  from keras.layers import LSTM,Merge,BatchNormalization,Dropout
  from keras.layers.merge import Concatenate
  from keras.layers.merge import Dot
  from keras.datasets import imdb
  from keras.layers import Flatten
  from keras.callbacks import EarlyStopping
  from sklearn.metrics import accuracy_score
  
  model = Sequential()
  model.add(Embedding(max_features, 16))
  model.add(LSTM(16, dropout=0.5, recurrent_dropout=0.1))  
  model.add(Dense(16, kernel_initializer='normal',activation='relu'))
  model.add(Dropout(0.5))
  model.add(BatchNormalization())
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy',optimizer='rmsprop')
  return model



from sklearn.model_selection import KFold
import numpy as np
foldCount = 3 
kf = KFold(n_splits=foldCount, random_state=52, shuffle=True)
fPreds = np.zeros(len(test)) 
print(fPreds.shape)
aucScore = 0

for train_index, test_index in kf.split(X_train):
    
  X_tr, X_te = X_train[train_index], X_train[test_index]
  y_tr, y_te = y_train[train_index], y_train[test_index]
  kmodel = KerasModel()
  print("Training ...")  
  kmodel.fit(X_tr, y_tr,
              batch_size=2048,
              epochs=15,
              validation_data=(X_te, y_te),
              class_weight=get_class_weights(y_tr, smooth_factor=0.15))  
  
  print("Predicting ...")
  valPred = kmodel.predict(X_te,batch_size=2048)
  valPred = valPred.flatten()
  valPredClass = kmodel.predict_classes(X_te,batch_size=2048)
  printConfMat(y_te,valPredClass)
  tempScore = auc(y_te,valPred)
  aucScore += tempScore
  print("tempScore",tempScore)
  preds = kmodel.predict(test,batch_size=2048)
  preds = preds.flatten()
  
     
  print(preds.shape)
  fPreds += preds

aucScore = aucScore/foldCount
fPreds = fPreds/foldCount
sub = pd.DataFrame({'ID':testID, 'click':fPreds})
sub.to_csv('KerasLSTM_3_%s.csv' %(str(aucScore)), index=False)


    
    
    
