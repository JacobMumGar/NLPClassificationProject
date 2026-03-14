from sklearn.model_selection import train_test_split

def trainTestSplit(df):
    X = df["transcript"]
    y = df["party"]
    
    trainX, trainY, testX, testY = train_test_split(X, y, test_size=0.2, random_state=123)

    return trainX, trainY, testX, testY