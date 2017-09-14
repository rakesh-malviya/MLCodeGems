import math
import numpy as np
"""
From tensorflow playground code
"""

def gen2Spiral(sample=500,noise=0,addSin=False):
    n = sample/2
    Xdata = []
    Ydata = []

    def genSpiral(deltaT,label):
        for i in range(n):
            r = float(i)/n*5.0
            t = 1.75*float(i)/n*2.0*math.pi + deltaT
            x = r*math.sin(t) #+ np.random.uniform(-1,1)*noise
            y = r * math.cos(t)  # + np.random.uniform(-1,1)*noise


            if addSin:
                Xdata.append([x, y,math.sin(x),math.sin(y)])
            else:
                Xdata.append([x, y])

            Ydata.append(label)


    genSpiral(0,1.0)
    genSpiral(math.pi,0.0)

    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)
    return Xdata,Ydata

# X,y=gen2Spiral(addSin=True)

# from matplotlib import pyplot as plt
# X,y=gen2Spiral()
# print(X)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
#             s=25, edgecolor='k')
# plt.show()



