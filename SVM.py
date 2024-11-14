import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0

        y = np.where(y <= 0, -1, 1)

        for _ in range(self.num_iterations):
            linear_output = np.dot(X, self.w) + self.b
            condition = y * linear_output >= 1

            dw = np.zeros(num_features)
            db = 0

            for idx, x_i in enumerate(X):
                if not condition[idx]:
                    dw = dw + y[idx] * x_i
                    db = db + y[idx]

            self.w = self.w - (self.lr * (2 * self.lambda_param * self.w - dw))
            self.b = self.b - (self.lr * db)

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

def evaluate_model(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

def main():
    X = np.array([
        #class 1
        [1, 2],  
        [2, 3],  
        [3, 3],  
        #class -1
        [5, 5],
        [6, 8],
        [7, 7]
    ])

    y = np.array([1, 1, 1, 0, 0, 0])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #initialize model
    svm = SVM(learning_rate=0.001, lambda_param=0.01, num_iterations=1000)

    #train model
    svm.fit(X, y)

    predictions = svm.predict(X)
    print("Predictions:", predictions)

    predictions = np.where(predictions == -1, 0, 1)  # Convert predictions back to 0/1 for evaluation

    print("Model Evaluation:")
    evaluate_model(y, predictions)

if __name__ == "__main__":
    main()



    