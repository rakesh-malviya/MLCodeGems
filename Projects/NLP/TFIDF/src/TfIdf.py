import os
import preprocess
from math import log,sqrt


class TfIdf:
  
  def __init__(self,dirPath):
    self.vocabIndex = {} #Keeps index of each word
    self.docIndex = {} #Keeps index of each doc
    self.tfMatrix = [] #term frequrency for each document
    self.readDocDir(dirPath)
      
    print("Vocab Size",len(self.vocabIndex),self.curWordCount)
    print("Doc Size",len(self.docIndex),self.curDocCount)
    self.generateIdfAll() 
    self.generateTfIdfAll()
     
    
        
  def readDocDir(self,dirPath):
    self.curDocCount = 0
    self.curWordCount = 0
    """Read all docs in dirPath one by one"""
    for filename in os.listdir(dirPath):
      curDocId = self.curDocCount
      self.docIndex[filename] = curDocId
      self.curDocCount += 1
      self.tfMatrix.append([]);
      wordList = preprocess.getWordsFromFile(dirPath+"/"+filename)
      for word in wordList:
        word = word.lower()
        wordIndex = self.vocabIndex.get(word)
        if wordIndex==None:          
          wordIndex = self.curWordCount
          self.vocabIndex[word] = wordIndex
          self.curWordCount += 1
        
        curDocTfLen = len(self.tfMatrix[curDocId])
        if wordIndex > curDocTfLen -1:
          self.tfMatrix[curDocId] += [0 for _ in range(wordIndex - curDocTfLen + 1)]
                  
        self.tfMatrix[curDocId][wordIndex] += 1
    
    for elem in self.tfMatrix:
      l1 = len(elem)
      elem += [0 for _ in range(self.curWordCount - l1)]

  def generateIdfAll(self):
    # To save number of document which have word at index i
    self.docCountPerWord = [0 for _ in range(self.curWordCount)]
    for i in range(self.curWordCount):
      for j in range(self.curDocCount):
        if self.tfMatrix[j][i]!=0:
          self.docCountPerWord[i] += 1
          
    
    
    """
    Generate idf
    From wiki: inverse document frequency smooth
    log(1+N/nt)
    N = total number of docs
    nt = Number Of Documents with given term/word
    """ 
    epsilon = 0.000001 # to remove divide by zero error
    self.idfMatrix = []
    for i in range(self.curDocCount):
      self.idfMatrix.append([])
      for j in range(self.curWordCount):
        j_idf = log(1 + (self.curDocCount/(self.docCountPerWord[j]+epsilon)))
        self.idfMatrix[i].append(j_idf)
        
  def l2_normalizer(self,vec):
    epsilon = 0.000001 # to remove divide by zero error
    denom = sum([el**2 for el in vec])
    return [(el / sqrt(denom+epsilon)) for el in vec]
  
  def cosine_similarity(self,vec1,vec2):
    simi = 0
    for i in range(len(vec1)):
      simi += vec1[i]*vec2[i]
      
    return simi

        
  def generateTfIdfAll(self):
    self.tfIdfMatrix = []    
    for i in range(self.curDocCount):
      self.tfIdfMatrix.append([])
      for j in range(self.curWordCount):        
        tfidf_ij = self.tfMatrix[i][j]*self.idfMatrix[i][j]
        self.tfIdfMatrix[i].append(tfidf_ij)
        
    
        
    """Normalize tfidf vector"""
    for i in range(self.curDocCount):
      self.tfIdfMatrix[i] = self.l2_normalizer(self.tfIdfMatrix[i])
      
  def findSimilarDocFromFromCorpus(self,fileName,n=5):
    print "Finding documents similar to doc:",fileName
    curDocIndex = self.docIndex[fileName]
    
    if curDocIndex==None:
      print("Error Document:%s not in Corpus" %(fileName))
      return 
    
    curDocTfIdf = self.tfIdfMatrix[curDocIndex]
    simiScore = []
    for tempDocName,tempDocId in self.docIndex.items():
      tempDocTfIdf = self.tfIdfMatrix[tempDocId]
      tempScore = self.cosine_similarity(curDocTfIdf, tempDocTfIdf)
      simiScore.append([tempDocName,tempScore])
      
    simiScore = sorted(simiScore, key=lambda x: x[1],reverse=True)

#     print "Document Similarity Scores",simiScore
    return [x for x,y in simiScore[:n]]
    
  def findSimilarNewDoc(self,docPath,n=5):
    print "Finding documents similar to doc:",docPath
    epsilon = 0.000001
    wordList = preprocess.getWordsFromFile(docPath)
    
    tfList = [0 for _ in range(self.curWordCount)]
    
    for word in wordList:
      word = word.lower()
      wordIndex = self.vocabIndex.get(word)
      
      if wordIndex!=None:
        tfList[wordIndex] += 1
    
    tfIdfVec = []
    
    for i in range(self.curWordCount):
      curTf = tfList[i]
      if curTf==0:
        curTfIdf = 0
      else:
        N = self.curDocCount + 1
        nt = self.docCountPerWord[i]
        if curTf>0:
          nt += 1
        
        curIdf = log(1 + (N/(nt+epsilon)))
        curTfIdf = curTf*curIdf
      
      tfIdfVec.append(curTfIdf)
      
    tfIdfVec = self.l2_normalizer(tfIdfVec)
    simiScore = []
    for tempDocName,tempDocId in self.docIndex.items():
      tempDocTfIdf = self.tfIdfMatrix[tempDocId]
      tempScore = self.cosine_similarity(tfIdfVec, tempDocTfIdf)
      simiScore.append([tempDocName,tempScore])
      
    simiScore = sorted(simiScore, key=lambda x: x[1],reverse=True)

#     print "Document Similarity Scores",simiScore
    return [x for x,y in simiScore[:n]]