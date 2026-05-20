from src.dataHandling.loadData import loadData
from src.dataHandling.dataPreprocessing import preprocess, chunkify, trainTestSplit
from src.embeddings.embeddings import TFIDF, word2vec, sentenceTransform
from src.models.models import logRegModel, svmModel, rfModel, mlpModel
from src.evaluation.evaluate import accuracy, confusionMatrix, classReport
import matplotlib.pyplot as plt 

#load the raw data from the data folder
df = loadData()

#preprocess the raw data
df["transcript"] = df["transcript"].apply(preprocess)

#split the data into train and test
dfTrain, dfTest = trainTestSplit(df)
trainX = dfTrain["transcript"]
testX = dfTest["transcript"]
trainY = dfTrain["party"]
testY = dfTest["party"]

#embed the data with each method
#tfidfTrain, tfidfTest = TFIDF(trainX, testX)
#word2vecTrain, word2vecTest = word2vec(trainX, testX)


sentTransTrain, sentTransTest = sentenceTransform(trainX, testX)

'''
#train each model with each embedding
#-----------------------TF IDF-----------------------#
lrTFIDFmodel = logRegModel(tfidfTrain, trainY)
svmTFIDFmodel = svmModel(tfidfTrain, trainY)
rfTFIDFmodel = rfModel(tfidfTrain, trainY)

#----------------------Word2Vec----------------------#
lrW2Vmodel = logRegModel(word2vecTrain, trainY)
svmW2Vmodel = svmModel(word2vecTrain, trainY)
rfW2Vmodel = rfModel(word2vecTrain, trainY)
'''

#----------------Sentence Transformer----------------#
lrSTmodel = logRegModel(sentTransTrain, trainY)
svmSTmodel = svmModel(sentTransTrain, trainY)
mlpSTmodel = mlpModel(sentTransTrain, trainY)

#evaluate each model
print(f"LogReg ST: {accuracy(lrSTmodel, sentTransTest, testY)}")
print(f"MLP ST: {accuracy(mlpSTmodel, sentTransTest, testY)}")
print(f"SVM ST: {accuracy(svmSTmodel, sentTransTest, testY)}")