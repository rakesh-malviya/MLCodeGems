import numpy as np



def load_word2vec(file):
  word2vec = {} #skip information on first line
  fin= open(file, encoding="utf8")    
  for line in fin:
              items = line.replace('\r','').replace('\n','').split(' ')
              if len(items) < 10: continue
              word = items[0]
              vectList  = [float(i) for i in items[1:] if len(i) >= 1]
              vectList.append(0.0)
              vect = np.array(vectList)                            
              word2vec[word] = vect
  
  
  return word2vec