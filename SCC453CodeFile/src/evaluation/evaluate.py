from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def accuracy(model, X, y):
    #make predictions on the test X values
    yPredictions = model.predict(X)

    #calculate the accuracy from the true values of y and the predicitons made by the model
    return accuracy_score(y, yPredictions)

def f1Score(model, X, y):
    yPredictions = model.predict(X)

    #calculate the f1 score from the true values of y and the predicitons made by the model
    return f1_score(y, yPredictions, pos_label="Republican")