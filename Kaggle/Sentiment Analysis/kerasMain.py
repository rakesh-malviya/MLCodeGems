import csv
import numpy as np
import utils

_unk_ = 0
_pad_ = 1
y_Encoder = utils.y_Encoder(['0','1','2','3','4'])
maxlen = 56
trainXCodeList = []
seqlenList = []
trainYCodeList = []
vocab = {}
vocab["_unk_"] = _unk_
vocab["_pad_"] = _pad_
header = True
next_index = 2
print("creating traindData and vocab...")
with open("data/train.tsv") as tsv:
  for line in csv.reader(tsv, dialect="excel-tab"):
    if header:
      header = False
      continue
    y_Val = line[3] 
    phrase = line[2]
    
    pwordList = phrase.split()
    seqlen = len(pwordList)
    #update vocab and generate phrase seq code
    seqcode = []
    for i in range(maxlen):
      try:
        word = pwordList[i]
        wordId = vocab.get(word)
        if wordId==None:
          vocab[word] = next_index
          wordId = next_index
          next_index+=1
        
        seqcode.append(wordId)
      except IndexError:
        seqcode.append(_pad_)
    
    trainXCodeList.append(np.array(seqcode))
    trainYCodeList.append(y_Encoder[y_Val])
    seqlenList.append(seqlen)

trainXData = np.stack(trainXCodeList, axis=0)
trainYData = np.stack(trainYCodeList, axis=0)
seqLenData = np.array(seqlenList)

print(trainXData.shape)
print(trainYData.shape)
print(seqLenData.shape)

from sklearn.model_selection import KFold
 
kf = KFold(n_splits=4, random_state=52, shuffle=True)

for train_index, test_index in kf.split(trainXData):
    
    X_train, X_test = trainXData[train_index], trainXData[test_index]
    y_train, y_test = trainYData[train_index], trainYData[test_index]
    seqlen_train, seqlen_test = seqLenData[train_index], seqLenData[test_index]
    
    print(X_train.shape)
    print(y_train.shape)
    print(seqlen_train.shape)
    break
  
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = len(vocab)
batch_size = 200
print(max_features,batch_size)
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.1))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc) #63,



