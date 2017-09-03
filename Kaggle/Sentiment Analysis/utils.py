import numpy as np

# unique class values
def y_Encoder(classList):
  num_class = len(classList)
  classDict = {}
  
  for i,classElem in enumerate(classList):
    code = np.zeros((num_class,))
    code[i] = 1.0
    classDict[classElem] = code
    
  return classDict

# from nltk.corpus import wordnet as wn
# nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
# print (nouns)

# checkList = ["multiplex",
#              "1984",
#              "tea",             
#              ]
# 
# sysn = wn.synsets("undercuts")
# 
# print (sysn[0])
  
  