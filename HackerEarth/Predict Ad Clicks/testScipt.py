import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
X_train, X_test, y_train, y_test = train_test_split(train[cols_to_use], train['click'], test_size = 0.5)

print("Training data.....")



dtrain = lgb.Dataset(X_train, y_train)
dval = lgb.Dataset(X_test, y_test)

params = {
    
    'num_leaves' : 256,
    'learning_rate':0.03,
    'metric':'auc',
    'objective':'binary',
    'early_stopping_round': 40,
    'max_depth':10,
    'bagging_fraction':0.5,
    'feature_fraction':0.6,
    'bagging_seed':2017,
    'feature_fraction_seed':2017,
    'verbose' : 1
}

clf = lgb.train(params, dtrain,num_boost_round=500,valid_sets=dval,verbose_eval=20)

preds = clf.predict(test[cols_to_use])

sub = pd.DataFrame({'ID':test['ID'], 'click':preds})
sub.to_csv('lgb_pyst.csv', index=False)


    
    
    
