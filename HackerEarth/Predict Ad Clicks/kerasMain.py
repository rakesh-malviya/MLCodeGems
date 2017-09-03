import pandas as pd
import numpy as np
# import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn import metrics
from keras import backend as K
from sklearn import preprocessing


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
# X_train = X_train[:100]
# y_train = y_train[:100]
testID = test['ID']
test = test[cols_to_use].values
print(X_train.shape,y_train.shape)
print("Training data.....")


def auc(y_true, y_pred):
    return metrics.roc_auc_score(K.eval(y_true), K.eval(y_pred))

def KerasModel():
  from keras.preprocessing import sequence
  from keras.models import Sequential
  from keras.layers import Dense, Embedding,Activation
  from keras.layers import LSTM,Merge,BatchNormalization
  from keras.layers.merge import Concatenate
  from keras.layers.merge import Dot
  from keras.datasets import imdb
  from keras.layers import Flatten
  from keras.callbacks import EarlyStopping
  from sklearn.metrics import accuracy_score
  
  model = Sequential()  
  model.add(Dense(50,input_dim=10, kernel_initializer='normal',activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(10, kernel_initializer='normal', activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy',optimizer='rmsprop')
  return model



from sklearn.model_selection import KFold
import numpy as np
foldCount = 10 
kf = KFold(n_splits=foldCount, random_state=52, shuffle=True)
fPreds = np.zeros(len(test)) 
print(fPreds.shape)
for train_index, test_index in kf.split(X_train):
    
  X_tr, X_te = X_train[train_index], X_train[test_index]
  y_tr, y_te = y_train[train_index], y_train[test_index]
  kmodel = KerasModel()
  print("Training ...")
  kmodel.fit(X_tr, y_tr,
              batch_size=2048,
              epochs=15,
              validation_data=(X_te, y_te))
  
  print("Predicting ...")
  preds = kmodel.predict(test,batch_size=2048)
  preds = preds.flatten() 
  print(preds.shape)
  fPreds += preds


fPreds = fPreds/foldCount
sub = pd.DataFrame({'ID':testID, 'click':fPreds})
sub.to_csv('lgb_pyst_Keras_1.csv', index=False)


    
    
    
