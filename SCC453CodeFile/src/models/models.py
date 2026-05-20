from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def logRegModel(X, y):
    #fit the logistic regression model on the training data
    model = LogisticRegression().fit(X, y)
    return model

def svcModel(X, y):
    #fit the svc model on the training data, random state used for reproducible results
    model = LinearSVC(random_state=123, max_iter=5000).fit(X, y)
    return model

def rfModel(X, y):
    #fit the random forest classifier, random state used for reproducible results
    # max_depth and min_sample_leaf both prevent overfitting
    model = RandomForestClassifier(random_state=123,n_estimators=300, max_depth=10, min_samples_leaf=2).fit(X, y)
    return model

def mlpModel(X, y):
    #fit the multi layer perceptron model, random state used for reproducible results
    #set the hidden layer of 64 neurons, just one layer to avoid overfitting,
    #adaptive learning rate is ideal for smaller datasets
    model = MLPClassifier(max_iter=300,learning_rate="adaptive", hidden_layer_sizes=(64,), random_state=123).fit(X, y)
    return model