import matplotlib.pyplot as plt
def drawBars(mapData,colStr):
    
    x = [i for i in mapData.keys()]
    y = [i for i in mapData.values()]
    
    width = 1/1.5
    plt.bar(x, y, width, color=colStr,alpha = 0.5)
    plt.show()