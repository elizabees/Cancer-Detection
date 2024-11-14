import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from math import sqrt
from collections import Counter
import os

#calculate distance
def calculate_distance(row1, row2, metric):
    if metric == 'euclidean':
        return sqrt(sum((x - y) ** 2 for x, y in zip(row1, row2)))
    elif metric == 'manhattan':
        return sum(abs(x - y) for x, y in zip(row1, row2))

#find nearest neighbors
def get_neighbors(train, test_point, num_neighbors, metric):
    distances = []
    for train_point in train:
        distance = calculate_distance(test_point, train_point[0], metric)
        distances.append((train_point, distance))
    
    distances.sort(key=lambda tup: tup[1])
    num_neighbors = min(num_neighbors, len(distances))
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors

#predict class by nearest neighbors
def predict_classification(train, test_point, num_neighbors, metric):
    neighbors = get_neighbors(train, test_point, num_neighbors, metric)
    neighbor_values = [row[-1] for row in neighbors]
    prediction = Counter(neighbor_values).most_common(1)[0][0]
    return prediction

def load_breast_cancer_data():
    data_path = os.path.join('data', 'wdbc.data')
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv(data_path, header=None, names=column_names)

    #convert 'Diagnosis' column to binary (M = malignant, B = benign)
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

    #drop ID column
    X = data.drop(['ID', 'Diagnosis'], axis=1).values
    y = data['Diagnosis'].values

    #shuffle and split the data
    np.random.seed(29)
    shuffled_indices = np.random.permutation(len(X))
    X, y = X[shuffled_indices], y[shuffled_indices]

    total_size = len(X)
    train_size = int(0.6 * total_size)  # 60% training
    val_size = int(0.2 * total_size)  # 20% validation
    test_size = total_size - train_size - val_size  # 20% testing

    #split train, validation, and test sets
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    #feature scaling (standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

#confusion matrix
def confusion_matrix(y_true, y_pred, num_classes):
    TP = [0] * num_classes
    TN = [0] * num_classes
    FP = [0] * num_classes
    FN = [0] * num_classes

    for true, pred in zip(y_true, y_pred):
        for class_index in range(num_classes):
            if true == class_index and pred == class_index:
                TP[class_index] += 1
            elif true != class_index and pred != class_index:
                TN[class_index] += 1
            elif true != class_index and pred == class_index:
                FP[class_index] += 1
            elif true == class_index and pred != class_index:
                FN[class_index] += 1

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    for class_index in range(num_classes):
        accuracy = (TP[class_index] + TN[class_index]) / (TP[class_index] + TN[class_index] + FP[class_index] + FN[class_index]) if (TP[class_index] + TN[class_index] + FP[class_index] + FN[class_index]) != 0 else 0
        precision = TP[class_index] / (TP[class_index] + FP[class_index]) if (TP[class_index] + FP[class_index]) != 0 else 0
        recall = TP[class_index] / (TP[class_index] + FN[class_index]) if (TP[class_index] + FN[class_index]) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    avg_accuracy = sum(accuracies) / num_classes
    avg_precision = sum(precisions) / num_classes
    avg_recall = sum(recalls) / num_classes
    avg_f1 = sum(f1_scores) / num_classes

    return avg_accuracy, avg_precision, avg_recall, avg_f1

#find best k
def find_best_k(X_train, y_train, X_val, y_val, metric):
    best_k = 1
    best_metrics = (0, 0, 0, 0)
    for k in range(1, 20):
        predictions = [predict_classification(list(zip(X_train, y_train)), point, num_neighbors=k, metric=metric) for point in X_val]
        accuracy, precision, recall, f1 = confusion_matrix(y_val, predictions, 2)
        best_accuracy, best_precision, best_recall, best_f1 = best_metrics
        if (
            accuracy > best_accuracy or
            (accuracy == best_accuracy and precision > best_precision) or
            (accuracy == best_accuracy and precision == best_precision and recall > best_recall) or
            (accuracy == best_accuracy and precision == best_precision and recall == best_recall and f1 > best_f1)
        ):
            best_k = k
            best_metrics = (accuracy, precision, recall, f1)

    best_accuracy, best_precision, best_recall, best_f1 = best_metrics
    print(f"Best k for {metric}: {best_k}, Accuracy: {best_accuracy:.2f}, Precision: {best_precision:.2f}, Recall: {best_recall:.2f}, F1-Score: {best_f1:.2f}")
    return best_k

#evaluate k-NN on the test set
def evaluate_knn(X_train, X_test, y_train, y_test, best_k, metric):
    predictions = [predict_classification(list(zip(X_train, y_train)), test_row, num_neighbors=best_k, metric=metric) for test_row in X_test]
    accuracy, precision, recall, f1 = confusion_matrix(y_test, predictions, 2)
    return accuracy, precision, recall, f1

def main():
    #load and split the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_breast_cancer_data()

    #find the best k
    best_k_euclidean = find_best_k(X_train, y_train, X_val, y_val, metric='euclidean')
    best_k_manhattan = find_best_k(X_train, y_train, X_val, y_val, metric='manhattan')

    #evaluate on the test set
    print("\nEvaluating Euclidean Distance with best k on test set:")
    acc_euclidean, prec_euclidean, rec_euclidean, f1_euclidean = evaluate_knn(X_train, X_test, y_train, y_test, best_k_euclidean, metric='euclidean')
    print(f"Best k: {best_k_euclidean}, Accuracy: {acc_euclidean:.2f}, Precision: {prec_euclidean:.2f}, Recall: {rec_euclidean:.2f}, F1-Score: {f1_euclidean:.2f}")

    print("\nEvaluating Manhattan Distance with best K on test set:")
    acc_manhattan, prec_manhattan, rec_manhattan, f1_manhattan = evaluate_knn(X_train, X_test, y_train, y_test, best_k_manhattan, metric='manhattan')
    print(f"Best k: {best_k_manhattan}, Accuracy: {acc_manhattan:.2f}, Precision: {prec_manhattan:.2f}, Recall: {rec_manhattan:.2f}, F1-Score: {f1_manhattan:.2f}")

if __name__ == "__main__":
    main()
