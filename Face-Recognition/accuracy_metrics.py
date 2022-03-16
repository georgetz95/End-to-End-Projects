def accuracy_metrics(classifier, X_train, X_test, y_train, y_test, y_pred, pred_type='classification', cm_totals=True, class_labels=None, plot_cm=False, return_cm=True):
    """
    A simple function that returns accuracy measurements for both classification and regression predictions.
    
    --- Parameters ---
    classifier (sklearn model) : The model that has been fitted to the dataset.
    
    X_train (DataFrame/Series/Array) : The training independent variables.
    
    X_test (DataFrame/Series/Array) : The testing independent variables.
    
    y_train (DataFrame/Series/Array) : The training dependent variable.
    
    y_test (DataFrame/Series/Array) : The testing dependent variable.
    
    y_pred (DataFrame/Series/Array) : The predicted values of the model.
    
    pred_type (str) : "classification" for classification predictions, "regression" for regression predictions.
    
    cm_totals (bool) : Whether the confusion matrix should have totals for both axes.
    
    class_labels (list) : The labels of the dependent variable.

    return_cm (bool) : Return the confusion matrix DataFrame instead of printing it.
    """
    import pandas as pd
    import numpy as np
    b1 = '='*60
    b2 = '-'*60
    
    if pred_type == 'classification':
        from sklearn.metrics import classification_report, confusion_matrix
        print(f"{b1}\nAccuracy(Train): {classifier.score(X_train, y_train)}")
        print(f"Accuracy(Test): {classifier.score(X_test, y_test)}\n{b1}")
        print(f"{b1}\nClassification Report:\n{classification_report(y_test, y_pred)}\n{b1}")
        cm = confusion_matrix(y_test, y_pred)
        
        if class_labels == None:
            #cm_df = pd.DataFrame(cm, columns=['Predicted ' + str(i) for i in range(len(cm))], index=['Actual ' + str(i) for i in range(len(cm))])
            cm_df = pd.DataFrame(cm, columns=[['Predicted'] * len(cm), [str(i) for i in range(len(cm))]], index=[['Actual'] * len(cm), [str(i) for i in range(len(cm))]])
        else:
            #cm_df = pd.DataFrame(cm, columns=['Predicted ' + name for name in class_labels], index=['Actual ' + class_labels[i] for i in range(len(class_labels))])
            cm_df = pd.DataFrame(cm, columns=[['Predicted'] * len(cm), class_labels], index=[['Actual'] * len(cm), class_labels])
            
        if cm_totals == True:
            cm_df['Predicted','Total'] = np.sum(cm_df, axis=1)
            cm_df.loc[('Actual','Total'),:] = np.sum(cm_df, axis=0)
        cm_df = cm_df.astype('int')
        
        if plot_cm == True:
            import itertools
            plt.figure(figsize=(8,6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix', fontsize=35)
            plt.colorbar()
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels, rotation=45, fontsize=15)
            plt.yticks(tick_marks, class_labels, fontsize=15)
            if normalize:
                cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontdict={'fontsize':18, 'fontweight':'semibold'})
            plt.tight_layout()
            plt.ylabel('True Label', fontsize=20)
            plt.xlabel('Predicted Label', fontsize=20)
            plt.show()
        
        if return_cm == True:
            return cm_df
            
    elif pred_type == 'regression':
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        msle = mean_squared_log_error(y_test, y_pred)
        rmsle = np.sqrt(msle)
        print(f"{b1}\n{b2}\nAccuracy(Train): {classifier.score(X_train, y_train)}")
        print(f"{b2}\nAccuracy(Test): {classifier.score(X_test, y_test)}")
        print(f"{b2}\nMean Squared Error: {mse}")
        print(f"{b2}\nRoot Mean Squared Error: {rmse}")
        print(f"{b2}\nMean Absolute Error: {mae}")
        print(f"{b2}\nMean Squared Log Error: {msle}")
        print(f"{b2}\nRoot Mean Squared Log Error: {rmsle}\n{b1}")