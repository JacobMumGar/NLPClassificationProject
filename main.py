from src.dataHandling.loadData import loadData
from src.dataHandling.dataPreprocessing import preprocess, trainTestSplit
from src.dataHandling.eda import wordCounts, graphs, speechLengths
from src.embeddings.embeddings import TFIDF, word2vec, sentenceTransform
from src.models.models import logRegModel, svcModel, rfModel, mlpModel
from src.evaluation.evaluate import accuracy, f1Score

#load the raw data from the data folder
df = loadData()

#preprocess the raw data
df["transcript"] = df["transcript"].apply(preprocess)

#do some eda on topics and speech length etc.
#generate stats about average speech length per party, and most common words across both and per party, plot on a bar chart
#commented out as it doesnt need to be printed for the model to work
'''
speechLengths(df) 

repRows = df[df["party"]=="Republican"]
demRows = df[df["party"]=="Democrat"]
rep10 = wordCounts(repRows)
dem10 = wordCounts(demRows)
total10 = wordCounts(df)

graphs(rep10,"Republican")
graphs(dem10,"Democrat")
'''

#split the data into train and test
dfTrain, dfTest = trainTestSplit(df)
trainX = dfTrain["transcript"]
testX = dfTest["transcript"]
trainY = dfTrain["party"]
testY = dfTest["party"]

#embed the data with each method, (TF-IDF, Word2Vec and SentenceTransformer) 
tfidfTrain, tfidfTest = TFIDF(trainX, testX)
word2vecTrain, word2vecTest = word2vec(trainX, testX)
sentTransTrain, sentTransTest = sentenceTransform(trainX, testX)


#train each model for each embedding (Logistic regression, Simple Vector Classifier, Random Forest and Multilayer Perceptron)
#-----------------------TF IDF-----------------------#
lrTFIDFmodel = logRegModel(tfidfTrain, trainY)
svmTFIDFmodel = svcModel(tfidfTrain, trainY)
rfTFIDFmodel = rfModel(tfidfTrain, trainY)
mlpTFIDFmodel = mlpModel(tfidfTrain, trainY)

#----------------------Word2Vec----------------------#
lrW2Vmodel = logRegModel(word2vecTrain, trainY)
svmW2Vmodel = svcModel(word2vecTrain, trainY)
rfW2Vmodel = rfModel(word2vecTrain, trainY)
mlpW2Vmodel = mlpModel(word2vecTrain, trainY)

#----------------Sentence Transformer----------------#
lrSTmodel = logRegModel(sentTransTrain, trainY)
svmSTmodel = svcModel(sentTransTrain, trainY)
rfSTmodel = rfModel(sentTransTrain, trainY)
mlpSTmodel = mlpModel(sentTransTrain, trainY)

#evaluate each model, computing the accuracy and F1 score for each combination 
#----------------------Accuracy---------------------------#
print(f"LogReg TF-IDF: {accuracy(lrTFIDFmodel, tfidfTest, testY)}")
print(f"SVM TF-IDF: {accuracy(svmTFIDFmodel, tfidfTest, testY)}")
print(f"RF TF-IDF: {accuracy(rfTFIDFmodel, tfidfTest, testY)}")
print(f"MLP TF-IDF: {accuracy(mlpTFIDFmodel, tfidfTest, testY)}")

print(f"LogReg W2V: {accuracy(lrW2Vmodel, word2vecTest, testY)}")
print(f"SVM W2V: {accuracy(svmW2Vmodel, word2vecTest, testY)}")
print(f"RF W2V: {accuracy(rfW2Vmodel, word2vecTest, testY)}")
print(f"MLP W2V: {accuracy(mlpW2Vmodel, word2vecTest, testY)}")

print(f"LogReg ST: {accuracy(lrSTmodel, sentTransTest, testY)}")
print(f"SVM ST: {accuracy(svmSTmodel, sentTransTest, testY)}")
print(f"RF ST: {accuracy(rfSTmodel, sentTransTest, testY)}")
print(f"MLP ST: {accuracy(mlpSTmodel, sentTransTest, testY)}")

#----------------------F1 Score--------------------------#
print(f"LogReg TF-IDF: {f1Score(lrTFIDFmodel, tfidfTest, testY)}")
print(f"SVM TF-IDF: {f1Score(svmTFIDFmodel, tfidfTest, testY)}")
print(f"RF TF-IDF: {f1Score(rfTFIDFmodel, tfidfTest, testY)}")
print(f"MLP TF-IDF: {f1Score(mlpTFIDFmodel, tfidfTest, testY)}")

print(f"LogReg W2V: {f1Score(lrW2Vmodel, word2vecTest, testY)}")
print(f"SVM W2V: {f1Score(svmW2Vmodel, word2vecTest, testY)}")
print(f"RF W2V: {f1Score(rfW2Vmodel, word2vecTest, testY)}")
print(f"MLP W2V: {f1Score(mlpW2Vmodel, word2vecTest, testY)}")

print(f"LogReg ST: {f1Score(lrSTmodel, sentTransTest, testY)}")
print(f"SVM ST: {f1Score(svmSTmodel, sentTransTest, testY)}")
print(f"RF ST: {f1Score(rfSTmodel, sentTransTest, testY)}")
print(f"MLP ST: {f1Score(mlpSTmodel, sentTransTest, testY)}")