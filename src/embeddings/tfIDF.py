from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

def TFIDF(train, test):
    #convert the text to TF-IDF feature vectors
    vectoriser = TfidfVectorizer(stop_words="english")
    tfidfTrain = vectoriser.fit_transform(train)
    tfidfTest = vectoriser.transform(test)

    return tfidfTrain, tfidfTest