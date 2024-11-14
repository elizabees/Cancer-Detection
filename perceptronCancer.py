import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os

def predict(row, weights):
    return np.dot(weights, row)

def update_weights(row, weights, label, learning_rate=0.1):
    return weights + learning_rate * label * np.array(row)

def train_perceptron(train, n_epoch, learning_rate=0.1):
    weights = np.zeros(train.shape[1] - 1)  #initialize weights
    for epoch in range(n_epoch):
        errors = 0
        for row in train:
            prediction = predict(row[:-1], weights) * row[-1]  # y * (w · x)
            if prediction <= 0:  #prediction incorrect
                weights = update_weights(row[:-1], weights, row[-1], learning_rate)
                errors += 1
        if errors == 0:
            break  #stop if no errors
    return weights


def main():
    #load data
    data_path = os.path.join('data', 'wdbc.data')
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv(data_path, header=None, names=column_names)

    #convert Diagnosis column to binary
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

    #drop ID column
    X = data.drop(['ID', 'Diagnosis'], axis=1).values
    y = data['Diagnosis'].values

    #split dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    #feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #prepare training data
    binary_y_train = np.where(y_train == 1, 1, -1)  # Convert 0 to -1
    train_data = np.column_stack((X_train, binary_y_train))

    #train
    weights = train_perceptron(train_data, n_epoch=1000)

    #predict on validation set
    y_pred_val = np.sign([predict(row, weights) for row in X_val])
    y_pred_val = np.where(y_pred_val == -1, 0, 1)  # Convert -1 back to 0

    #calculate performance metrics on validation set
    accuracy = accuracy_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val)
    recall = recall_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)

    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation Recall: {recall:.4f}')
    print(f'Validation F1-score: {f1:.4f}')

    #predict on test set
    y_pred_test = np.sign([predict(row, weights) for row in X_test])
    y_pred_test = np.where(y_pred_test == -1, 0, 1)  # Convert -1 back to 0

    #calculate performance metrics on test set
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    print(f'Test Accuracy: {accuracy_test:.4f}')
    print(f'Test Precision: {precision_test:.4f}')
    print(f'Test Recall: {recall_test:.4f}')
    print(f'Test F1-score: {f1_test:.4f}')

if __name__ == "__main__":
    main()
