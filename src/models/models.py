from sklearn.model_selection import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def logRegModel(X, y):
    #fit the logistic regression model
    model = LogisticRegression().fit(X,y)
    return model

def svmModel(X,y):
    #scale x for the model
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #fit the svc model
    model = LinearSVC().fit(X, y)
    return model

def rfModel(X,y):
    #fit the random forest classifier
    model = RandomForestClassifier().fit(X,y)
    return model