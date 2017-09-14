import spiralgen
from basicNN import NeuralNetwork
from sklearn.model_selection import train_test_split

Xdata,Ydata = spiralgen.gen2Spiral(addSin=True)
nn = NeuralNetwork(4,[4,2,1],lr=0.05)
# Xdata,Ydata = spiralgen.gen2Spiral()
# nn = NeuralNetwork(2,[8,8,5,1],lr=0.003)
epoch = 10000

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.30,random_state=1010)

print(X_train.shape,X_test.shape)

nn.train(epoch,X_train,y_train,X_test=X_test,y_test=y_test,
         monitor_train_acc=True,monitor_test_acc=True,monitor_checkpoint=10)
