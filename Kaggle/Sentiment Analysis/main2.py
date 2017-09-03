import glove
import csv
import numpy as np
import loadsave
import utils


w2vlen = 50
word2vec = glove.load_word2vec("data/glove.6B."+str(w2vlen)+"d.txt")
y_Encoder = utils.y_Encoder(['0','1','2','3','4']) 
w2vlen = w2vlen+1
_unk_ = np.zeros((w2vlen,))
_unk_[0] = 1.0
_unk_[-1] = 1.0
_pad_ = np.zeros((w2vlen,))
_pad_[1] = 1.0
_pad_[-1] = 1.0
maxlen = 56

def processSentence(sent):
  sent = sent.lower()
  return sent

def encodeSent(sent):
  sent = processSentence(sent)
  wordList = sent.split()
  codeList = []
  for i in range(maxlen):
    try:
      code = word2vec[wordList[i]]
    except KeyError:
      code = _unk_
    except IndexError:
      code = _pad_
    codeList.append(code)
  
  return np.stack(codeList, axis=0),len(wordList)  

trainXCodeList = []
seqlenList = []
trainYCodeList = []

header = True
with open("data/train.tsv") as tsv:
  for line in csv.reader(tsv, dialect="excel-tab"):
    if header:
      header = False
      continue
    y_Val = line[3] 
    seqcode,seqlen = encodeSent(line[2])
    trainXCodeList.append(seqcode)
    try:
      trainYCodeList.append(y_Encoder[y_Val])
    except:
      print (line)
      trainYCodeList.append(y_Encoder[y_Val])
    seqlenList.append(seqlen)
    
print("saving Train Data .... ")

trainXData = np.stack(trainXCodeList, axis=0)
trainYData = np.stack(trainYCodeList, axis=0)
seqLenData = np.array(seqlenList)
 
print(trainXData.shape)
print(trainYData.shape)
print(seqLenData.shape)

# trainYseq = np.concatenate([trainYData,np.reshape(seqLenData,(-1,1))],axis = 1)
# print(trainYseq.shape)

# loadsave.save_obj(trainData, "trainData")
# loadsave.save_obj(seqLenData, "seqLenData")
print("Train Data saved .... ")

'''=========================================================
              Tensorflow code
   =========================================================
'''

import tensorflow as tf
from tensorflow.contrib import rnn
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


# Import MNIST data
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.01
training_iters = 15
batch_size = 200
display_step = 10

# Network Parameters
n_input = w2vlen # MNIST data input (img shape: 28*28)
n_steps = maxlen # timesteps
n_hidden = 200 # hidden layer num of features
n_classes = 5 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
# seqlentf = tf.placeholder("float", [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
#     # Reshaping to (n_steps*batch_size, n_input)
#     x = tf.reshape(x, [-1, n_input])
#     # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#     x = tf.split(x, n_steps, 0)
#     x = tf.stack(x, axis=0)
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=0.5)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 3)
    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x,dtype=tf.float32,time_major=True)

    print(outputs)
    print(states)
    # Linear activation, using rnn inner loop last output
    
    
    outputs = tf.unstack(outputs,axis=0)
    print(outputs)
    print(len(outputs))
    
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']
    states = tf.reduce_sum(states, axis=0)
    return tf.matmul(states[-1], weights['out']) + biases['out']

pred = RNN(x,weights, biases)
# print(pred)
# exit()
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
print("Training started")
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    
    train_idlist = [i for i in range(X_train.shape[0])]
    
    for epoch in range(training_iters):
      start = 0
      while(start < X_train.shape[0]):    
        if start+batch_size < X_train.shape[0]:
          batch_index = train_idlist[start:start+batch_size]
        else:
          batch_index = train_idlist[start:X_train.shape[0]]        
    
        batch_index = np.random.choice(train_idlist,batch_size)
        batch_x, batch_y,batch_seqlen = X_train[batch_index],y_train[batch_index],seqlen_train[batch_index] 
        
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print(str(epoch)+" Iter" + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            

        step += 1
        start += batch_size
      print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images    
    print("Testing Accuracy:", \
       sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
    
#     test_size = 1000
#     test_pred = np.zeros()
#     start = 0
#     while(start+test_size < X_test.shape[0]):
#       tempPred = sess.run(correct_pred,feed_dict={x: X_test[start:start+test_size], y: y_test[start:start+test_size],seqlentf:seqlen_test[start:start+test_size]})     
    
#     test_len = 128
#     test_data = X_train[:test_len]
#     test_label = y_train[:test_len]
#     test_seqlen =  seqlen_train[:test_len]
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: test_data, y: test_label,seqlentf:test_seqlen}))



