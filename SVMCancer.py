import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000, tolerance=1e-3):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.w = None
        self.b = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0

        #convert labels to -1 and 1
        y = np.where(y <= 0, -1, 1)

        #gradient descent
        for iteration in range(self.num_iterations):
            #compute linear output for all samples
            linear_output = np.dot(X, self.w) + self.b
            #vectorized condition checking
            incorrect_classifications = y * linear_output < 1
            
            #update weights and bias with vectorized operations
            dw = -np.dot(X[incorrect_classifications].T, y[incorrect_classifications]) / num_samples + (2 * self.lambda_param * self.w)
            db = -np.sum(y[incorrect_classifications]) / num_samples
            
            #update weights and bias
            previous_w = np.copy(self.w)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            #check for convergence
            if np.linalg.norm(self.w - previous_w) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations.")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

def main():
    #load and preprocess data
    data_path = os.path.join('data', 'wdbc.data')
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv(data_path, header=None, names=column_names)

    #convert diagnosis column to binary
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
    X = data.drop(['ID', 'Diagnosis'], axis=1).values
    y = data['Diagnosis'].values

    #split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    #feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #initialize model w/ hyperparameters
    svm = SVM(learning_rate=0.001, lambda_param=0.01, num_iterations=1000)

    #train model
    svm.fit(X_train, y_train)

    #validation set evaluation
    val_predictions = svm.predict(X_val)
    val_predictions = np.where(val_predictions == -1, 0, 1)
    print("Validation Set Evaluation:")
    evaluate_model(y_val, val_predictions)

    #test set evaluation
    test_predictions = svm.predict(X_test)
    test_predictions = np.where(test_predictions == -1, 0, 1)
    print("\nTest Set Evaluation:")
    evaluate_model(y_test, test_predictions)

if __name__ == "__main__":
    main()