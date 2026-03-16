from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def accuracy(model, X, y):
    #make predictions on the test X values
    yPredictions = model.predict(X)

    #calculate the accuracy from the true values of y and the predicitons made by the model
    accuracy = accuracy_score(y, yPredictions)
    return accuracy

def classReport(model, X, y):
    yPredictions = model.predict(X)
    print(classification_report(y, yPredictions))

def confusionMatrix(model, X, y):
    #make predictions on the test X values
    yPredictions = model.predict(X)
    #create a confusion matrix
    print(confusion_matrix(y, yPredictions))