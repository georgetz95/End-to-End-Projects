class results_dataframe:
    """
    prediction_type (str): Either 'regression' or 'classification' depending on the output type.
    model_name (str): The name of the model that made the predictions.
    y_true (pd.Series or list): The Actual values.
    y_pred (pd.Series or list): The predicted values.
    """
    def __init__(self, prediction_type):
        self.prediction_type = prediction_type
        if self.prediction_type == 'regression':
            self.results = pd.DataFrame({'Model': [], '$R^{2}$': [], 
                                         'Adjusted $R^{2}$': [], 'MSE': [], 'RMSE': [], 'MAE': [], 'RMAE': []})
        elif self.prediction_type == 'classification':
            self.results = pd.DataFrame({'Model': [], 'Accuracy': [],'Precision': [], 'True Pos. Rate(Recall)': [], 
                                         'True Neg. Rate': [], 'False Pos. Rate': [], 'False Neg. Rate': [], 'F1-Score': []})
        
    def append_results(self, model_name, y_true, y_pred):
        if self.prediction_type == 'regression':
            r2 = r2_score(y_true, y_pred)
            n = len(y_true)
            p = len(X_train.columns)
            adjusted_r2 = 1 - ((1 - r2)* (n - 1) / (n - p - 1))
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            rmae = np.sqrt(mae)

            vals = [model_name, r2, adjusted_r2, mse, rmse, mae, rmae]

        
        elif self.prediction_type == 'classification':
            TP = np.where((y_true ==1) &  (y_pred==1), 1, 0).sum()
            TN = np.where((y_true ==0) &  (y_pred==0), 1, 0).sum()
            FP = np.where((y_true ==0) &  (y_pred==1), 1, 0).sum()
            FN = np.where((y_true ==1) &  (y_pred==0), 1, 0).sum()
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            true_pos_recall = recall_score(y_true, y_pred)
            true_neg = TN / (TN + FP)
            false_pos = FP / (FP + TN)
            false_neg = FN / (FN + TP)
            f1 = f1_score(y_true, y_pred)
            
            vals = [model_name, accuracy, precision, true_pos_recall, true_neg, false_pos, false_neg, f1]
            
        cols = list(self.results.columns)
        new_row = dict()

        for var, val in zip(cols, vals):
            new_row[var] = val

        self.results = self.results.append(new_row, ignore_index=True)
        return self.results
            
            
    
    def remove_row(self, row_number, reset_index=True):
        self.results = self.results.drop(row_number, axis=0)
        if reset_index:
            self.results = self.results.reset_index()
        return self.results
        
        
    
