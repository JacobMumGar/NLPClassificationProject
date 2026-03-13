from src.dataHandling.loadData import loadData
from src.dataHandling.dataPreprocessing import preprocess
from src.dataHandling.trainTestSplit import trainTestSplit


#load the raw data from the data folder
df = loadData()

#preprocess the raw data
df["transcript"] = df["transcript"].apply(preprocess)

#split the data into train and test
trainX, trainY, testX, testY = trainTestSplit(df)

#embed the data with each method


#train each model with each embedding

#evaluate each model