from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def logRegModel(X, y):
    #fit the logistic regression model
    model = LogisticRegression().fit(X,y)
    return model

def svmModel(X,y):
    #fit the svc model
    model = LinearSVC().fit(X, y)
    return model

def rfModel(X,y):
    #fit the random forest classifier
    model = RandomForestClassifier().fit(X,y)
    return model