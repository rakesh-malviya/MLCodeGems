import csv
import numpy as np
import utils
import matplotlib.pyplot as plt
from pprint import pprint
import operator
# from nltk.corpus import wordnet as wn
# nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

_unk_ = 0
_pad_ = 1
y_Encoder = utils.y_Encoder(['0','1','2','3','4'])
maxlen = 56
trainXCodeList = []
trainXPhraseList = []
seqlenList = []
trainYCodeList = []
vocab = {}
freqDict = {}
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
    phrase = line[2].lower()
    
    pwordList = phrase.split()
    for word in pwordList:
      wordCount = freqDict.get(word)
      if wordCount == None:
        wordCount = 1
      else:
        wordCount += 1
      freqDict[word] = wordCount
        
    seqlen = len(pwordList)    
    trainXPhraseList.append(pwordList)    
    trainYCodeList.append(y_Encoder[y_Val])
    seqlenList.append(seqlen)



# D = freqDict
# plt.bar(range(len(D)), D.values(), align='center')
# # plt.xticks(range(len(D)), D.keys())
# plt.show()

sorted_x = sorted(freqDict.items(), key=operator.itemgetter(1))
minFreq = 10
#create vocab


for pair in sorted_x:
  word,count = pair  
  if count < minFreq:
#       print (pair)   
#     if word in nouns:
#       print ("w",word)
#       vocab[word] = _unk_ 
      continue
  
  vocab[word] = next_index
  next_index+=1

print("Vocab Len",len(vocab))

print ("creating vocab")
for pwordList in trainXPhraseList:
  pCodeList = []
  for i in range(maxlen):    
    try:
      word = pwordList[i]
      pCodeList.append(vocab[word])
    except KeyError:
      pCodeList.append(_unk_)
#       print("Error",word)
    except IndexError:
      pCodeList.append(_pad_)
      
  trainXCodeList.append(np.array(pCodeList))

print ("vocab done")

trainXData = np.stack(trainXCodeList, axis=0)
trainYData = np.stack(trainYCodeList, axis=0)
seqLenData = np.array(seqlenList)

print(trainXData.shape)
print(trainYData.shape)
print(seqLenData.shape)

from sklearn.model_selection import KFold
 
kf = KFold(n_splits=4, random_state=52, shuffle=True)

for train_index, test_index in kf.split(trainXData):
    
#     X_train, X_test = trainXData[train_index], trainXData[test_index]
#     y_train, y_test = trainYData[train_index], trainYData[test_index]
#     seqlen_train, seqlen_test = seqLenData[train_index], seqLenData[test_index]
    
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


testXCodeList = []
phraseIdList = []

header = True
with open("data/test.tsv") as tsv:
  for line in csv.reader(tsv, dialect="excel-tab"):
    if header:
      header = False
      continue 
    phrase = line[2].lower()
    phraseId = line[0]
    phraseIdList.append(phraseId)
    pwordList = phrase.split()
    pCodeList = []
    for i in range(maxlen):    
      try:
        word = pwordList[i]
        pCodeList.append(vocab[word])
      except KeyError:
        pCodeList.append(_unk_)
  #       print("Error",word)
      except IndexError:
        pCodeList.append(_pad_)
        
    testXCodeList.append(np.array(pCodeList))

testXData = np.stack(testXCodeList, axis=0)
test_y = model.predict_classes(testXData, batch_size=200)
print (len(phraseIdList),len(test_y))
import pandas as pd
data = {
    "PhraseId":phraseIdList,"Sentiment":test_y
  }

df = pd.DataFrame.from_dict(data,dtype=np.int64)

df.to_csv("output.csv",index = False)



