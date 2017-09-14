from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np
from activation import Sigmoid,Tanh

class Dense:
    def __init__(self,layer_size,input_size,act_func=None):
        self.layer_size = layer_size
        self.input_size = input_size
        self.W = np.random.normal(loc=0,scale=1.0,size=(self.layer_size,self.input_size))
        self.b = np.zeros(shape=(self.layer_size,1)) + 0.1
        self.h = None #output
        self.z = None
        self.grad_act_z = None
        self.delta = None
        self.grad_w = np.zeros(shape=self.W.shape)
        self.grad_b = np.zeros(shape=self.b.shape)
        self.grad_w_count = 0
        self.grad_b_count = 0


        if act_func!=None:
            self.act_func = act_func
        else:
            self.act_func = Tanh()

    def eval_z(self,input_h):

        #Test input shape
        if input_h.shape[0]!=self.input_size or input_h.shape[1]!=1:
            raise ValueError("input shape : %s expected (%d,1)"
                             %(str(input_h.shape),self.input_size))

        self.z = np.dot(self.W,input_h) + self.b

        # Test z shape
        if self.z.shape[0] != self.layer_size or self.z.shape[1] != 1:
            raise ValueError("input shape : %s expected (%d,1)"
                             % (str(self.z.shape), self.layer_size))

        self.grad_act_z = self.act_func.eval_grad(self.z)

    def eval_h(self,input_h):
        self.eval_z(input_h)
        self.h = self.act_func.eval(self.z)

    def set_delta(self,delta):
        self.delta = delta

    """
    delta^(l-1) = [ (W^l)^T delta^l ] grad(act(z^(l-1)))
    """
    def eval_delta_back(self,grad_act_z_back):
        if self.delta is None:
            raise ValueError("self.delta is None")

        if self.grad_act_z is None:
            raise ValueError("self.grad_act_z is None")

        return np.dot(np.transpose(self.W),self.delta) * grad_act_z_back

    def accum_grad(self,input_h):
        self.grad_w += np.matmul(self.delta, np.transpose(input_h))
        self.grad_b += self.delta
        self.grad_w_count += 1
        self.grad_b_count += 1


    def update_w(self,lr):
        self.W = self.W - self.grad_w*lr/self.grad_w_count
        self.b = self.b - self.grad_b*lr/self.grad_b_count
        self.grad_w = np.zeros(shape=self.W.shape)
        self.grad_b = np.zeros(shape=self.b.shape)
        self.grad_w_count = 0
        self.grad_b_count = 0


class NeuralNetwork:
    def __init__(self,input_size,layers,lr=0.01,batch_size=10):
        self.input_size = input_size
        self.lr = lr
        self.layers = []
        self.train_step_count = 0
        self.batch_size = batch_size

        cur_input_size = self.input_size
        for cur_layer_size in layers[:-1]:
            self.layers.append(Dense(cur_layer_size,cur_input_size))
            cur_input_size = cur_layer_size


        self.layers.append(Dense(layers[-1],cur_input_size,act_func=Sigmoid()))

        print([x.layer_size for x in self.layers])


    def forward_step(self,input_x):
        cur_input_h = input_x
        # print(self.layers[1].W)
        for layer in self.layers:
            layer.eval_h(cur_input_h)
            cur_input_h = layer.h

        return cur_input_h

    def backprop_step(self,input_x):
        rev_layer_idxes = list(range(len(self.layers)))[:-1]
        rev_layer_idxes.reverse()
        # print(rev_layer_idxes)
        # backpropagation
        for layer_idx in rev_layer_idxes:
            prev_layer = self.layers[layer_idx+1]
            layer = self.layers[layer_idx]
            prev_layer.accum_grad(layer.h)
            cur_delta = prev_layer.eval_delta_back(layer.grad_act_z)
            layer.set_delta(cur_delta)

        layer.accum_grad(input_x)

    def train_step(self,input_x,y):
        #forward step
        pred_y = self.forward_step(input_x=input_x)

        #Cost grading
        y = np.array(y)
        y = y.reshape((-1,1))
        cur_delta = (pred_y-y)*self.layers[-1].grad_act_z
        self.layers[-1].set_delta(cur_delta)

        #backpropagation step
        self.backprop_step(input_x)
        self.train_step_count+=1
        if self.train_step_count%self.batch_size==0:
            self.updateW()


    def updateW(self):
        for layer in self.layers:
            layer.update_w(self.lr)

    def predict(self,X_train,y_train):
        pred_y = []
        pred_pro_y = []
        for j in range(X_train.shape[0]):
            X = X_train[j, :]
            X = X.reshape((-1, 1))
            y = y_train[j]
            y = y.reshape((-1, 1))
            y_p = self.forward_step(X)
            pred_pro_y.append(y_p[0])
            if y_p[0] > 0.5:
                pred_y.append(1)
            else:
                pred_y.append(0)

        pred_y = np.array(pred_y)
        pred_pro_y = np.array(pred_pro_y)

        return pred_y,pred_pro_y


    def train(self,epoch,X_train,y_train,X_test=None,y_test=None,
              monitor_train_acc=False,monitor_test_acc=False,monitor_checkpoint=10):
        for iepo in range(epoch):
            for j in range(X_train.shape[0]):
                X = X_train[j, :]
                X = X.reshape((-1, 1))
                y = y_train[j]
                y = y.reshape((-1, 1))
                self.train_step(X, y)


            if (iepo % monitor_checkpoint == 0):
                if monitor_train_acc and X_train is not None and y_train is not None:
                    pred_y, pred_pro_y = self.predict(X_train,y_train)
                    print("%d Train acc:%f mse:%f lr:%f" % (
                    iepo, accuracy_score(y_train, pred_y), mean_squared_error(y_train, pred_pro_y), self.lr))

                if monitor_test_acc and X_test is not None and y_test is not None:
                    pred_y, pred_pro_y = self.predict(X_test, y_test)
                    print("%d Test acc:%f mse:%f lr:%f" % (
                    iepo, accuracy_score(y_test, pred_y), mean_squared_error(y_test, pred_pro_y), self.lr))





