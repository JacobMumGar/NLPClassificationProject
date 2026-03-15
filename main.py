from src.dataHandling.loadData import loadData
from src.dataHandling.dataPreprocessing import preprocess
from src.dataHandling.trainTestSplit import trainTestSplit
from src.embeddings.tfIDF import TFIDF
from src.models.models import logRegModel, svmModel, rfModel
from src.evaluation.evaluate import evaluate

#load the raw data from the data folder
df = loadData()

#preprocess the raw data
df["transcript"] = df["transcript"].apply(preprocess)

#split the data into train and test
trainX, trainY, testX, testY = trainTestSplit(df)

#embed the data with each method
embeddedTrain, embeddedTest = TFIDF(trainX, testX)

#train each model with each embedding
LRmodel = logRegModel()
SVMmodel = svmModel()
RandomForestModel = rfModel()

#evaluate each model
evaluate(LRmodel, embeddedTest, testY)
evaluate(SVMmodel, embeddedTest, testY)
evaluate(RandomForestModel, embeddedTest, testY)