from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate(model, X, y):
    #make predictions on the test X values
    yPredictions = model.predict(X)

    #calculate the accuracy from the true values of y and the predicitons made by the model
    accuracy = accuracy_score(y, yPredictions)

    #create a confusion matrix
    print(accuracy)