from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
import numpy as np

def TFIDF(train, test):
    #convert the text to TF-IDF feature vectors
    vectoriser = TfidfVectorizer(stop_words="english")
    tfidfTrain = vectoriser.fit_transform(train)
    tfidfTest = vectoriser.transform(test)

    return tfidfTrain, tfidfTest

def word2vec(train, test):
    #tokenise the speeches
    trainTokens = [nltk.word_tokenize(speech) for speech in train]
    testTokens = [nltk.word_tokenize(speech) for speech in test]

    #train the word2vec model on the training data
    model = Word2Vec(sentences=trainTokens, window=5, vector_size=100, min_count=1)

    #words in the speeches are converted to word2vec 100 dimensional vectors, then for each speech
    # we take the average of all the word vectors
    w2vTrain = []
    w2vTest = []
    for speech in trainTokens:
        speechVectors = [model.wv[word] for word in speech if word in model.wv]
        w2vTrain.append(np.mean(speechVectors, axis=0))

    for speech in testTokens:
        speechVectors = [model.wv[word] for word in speech if word in model.wv]
        if speechVectors:
            w2vTest.append(np.mean(speechVectors, axis=0))
        else:
            w2vTest.append(np.zeros(100))
    
    return w2vTrain, w2vTest

def sentenceTransform(train, test):
    return