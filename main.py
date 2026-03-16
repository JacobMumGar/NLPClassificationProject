from src.dataHandling.loadData import loadData
from src.dataHandling.dataPreprocessing import preprocess, chunkify, trainTestSplit
from src.embeddings.embeddings import TFIDF, word2vec, sentenceTransform
from src.models.models import logRegModel, svmModel, rfModel
from src.evaluation.evaluate import accuracy, confusionMatrix, classReport

#load the raw data from the data folder
df = loadData()

#preprocess the raw data
df["transcript"] = df["transcript"].apply(preprocess)

#split the data into train and test
dfTrain, dfTest = trainTestSplit(df)

#split the data into chunks
trainChunks = chunkify(dfTrain,100)
trainX = [chunk[0] for chunk in trainChunks]
trainY = [chunk[1] for chunk in trainChunks]

testChunks = chunkify(dfTest,100)
testX = [chunk[0] for chunk in testChunks]
testY = [chunk[1] for chunk in testChunks]

print(len(trainX))
print(len(testX))

'''
#embed the data with each method
tfidfTrain, tfidfTest = TFIDF(trainX, testX)
word2vecTrain, word2vecTest = word2vec(trainX, testX)
#sentTransTrain, sentTransTest = sentenceTransform(trainX, testX)

#train each model with each embedding
logistic_regression_model2 = logRegModel(tfidfTrain, trainY)
simple_vector_machine_model2 = svmModel(tfidfTrain, trainY)
random_forest_model2 = rfModel(tfidfTrain, trainY)
logistic_regression_model = logRegModel(word2vecTrain, trainY)
simple_vector_machine_model = svmModel(word2vecTrain, trainY)
random_forest_model = rfModel(word2vecTrain, trainY)

#evaluate each model
lrAcc2 = accuracy(logistic_regression_model2, tfidfTest, testY)
svmAcc2 = accuracy(simple_vector_machine_model2, tfidfTest, testY)
rfAcc2 = accuracy(random_forest_model2, tfidfTest, testY)
print("="*50)
print("TF-IDF")
print(f"Logistic Regression: {lrAcc2:.3f}")
print(f"Simple Vector Machine: {svmAcc2:.3f}")
print(f"Random Forest: {rfAcc2:.3f}")
print("="*50)
lrAcc = accuracy(logistic_regression_model, word2vecTest, testY)
svmAcc = accuracy(simple_vector_machine_model, word2vecTest, testY)
rfAcc = accuracy(random_forest_model, word2vecTest, testY)
print("="*50)
print("Word2Vec")
print(f"Logistic Regression: {lrAcc:.3f}")
print(f"Simple Vector Machine: {svmAcc:.3f}")
print(f"Random Forest: {rfAcc:.3f}")
print("="*50)'''