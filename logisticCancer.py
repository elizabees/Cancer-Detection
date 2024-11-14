import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os #data loading

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000, l2_reg=0.0):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg  # L2 regularization for MAP
        self.weights = None
        self.bias = None

    #to output values between 0 and 1
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent for MLE/MAP
        for epoch in range(self.n_epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            #calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.l2_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            #ppdate weights/bias
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    #predict probability of class being 1
    def predict_prob(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    #predict binary class (0 or 1)
    def predict(self, X):
        y_pred = self.predict_prob(X)
        return [1 if i > 0.5 else 0 for i in y_pred]

def evaluate_model(y_true, y_pred, dataset_name=""):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} Precision: {precision:.4f}")
    print(f"{dataset_name} Recall: {recall:.4f}")
    print(f"{dataset_name} F1-score: {f1:.4f}")

def main():
    #load data
    data_path = os.path.join('data', 'wdbc.data')
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv(data_path, header=None, names=column_names)

    #convert diagnosis column to binary (M = malignant, B = benign)
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

    #drop ID column
    X = data.drop(['ID', 'Diagnosis'], axis=1).values
    y = data['Diagnosis'].values

    #split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    #feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #logistic regression with MLE (no regularization)
    print("\nLogistic Regression (MLE)")
    model_mle = LogisticRegression(learning_rate=0.01, n_epochs=1000, l2_reg=0.0)
    model_mle.fit(X_train, y_train)

    #predict and evaluate on the validation set (mle)
    y_val_pred_mle = model_mle.predict(X_val)
    evaluate_model(y_val, y_val_pred_mle, "Validation (MLE)")

    #predict and evaluate on the test set (mle)
    y_test_pred_mle = model_mle.predict(X_test)
    evaluate_model(y_test, y_test_pred_mle, "Test (MLE)")

    #logistic regression with MAP (L2 regularization)
    print("\nLogistic Regression (MAP - L2 Regularization)")
    model_map = LogisticRegression(learning_rate=0.01, n_epochs=1000, l2_reg=0.1)  # Adding L2 regularization
    model_map.fit(X_train, y_train)

    #predict and evaluate on the validation set (map)
    y_val_pred_map = model_map.predict(X_val)
    evaluate_model(y_val, y_val_pred_map, "Validation (MAP)")

    #predict and evaluate on the test set (map)
    y_test_pred_map = model_map.predict(X_test)
    evaluate_model(y_test, y_test_pred_map, "Test (MAP)")

if __name__ == "__main__":
    main()