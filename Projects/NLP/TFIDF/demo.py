"""Build model"""    
from src.TfIdf import TfIdf

"""Generate Corpus by given document dir path"""
tfidfModel = TfIdf("poem")

"""Get n most similar docs in corpus given name of the document in the corpus"""
print tfidfModel.findSimilarDocFromFromCorpus('16',n=5)

"""Get n most similar docs in corpus given path to a new document"""
print tfidfModel.findSimilarNewDoc("poem/16",n=5)