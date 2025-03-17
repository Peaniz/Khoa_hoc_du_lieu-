import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist

def sigmoid(x):
    """
    Hàm sigmoid cho logistic regression
    """
    return 1 / (1 + np.exp(-x))

def logistic_regression(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Thực hiện logistic regression từ đầu
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(n_iterations):
        # Forward pass
        linear_pred = np.dot(X, weights) + bias
        predictions = sigmoid(linear_pred)
        
        # Backward pass
        dw = (1/n_samples) * np.dot(X.T, (predictions - y))
        db = np.sum(predictions - y)
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
    
    return weights, bias

def decision_tree(X, y, max_depth=5):
    """
    Thực hiện decision tree từ đầu
    """
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def gini(y):
        classes, counts = np.unique(y, return_counts=True)
        N = len(y)
        gini = 1.0
        for count in counts:
            gini -= (count/N) ** 2
        return gini
    
    def split(X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
    
    def find_best_split(X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gini_left = gini(y_left)
                gini_right = gini(y_right)
                gini_split = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
                
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(X, y, depth=0):
        if depth >= max_depth or len(np.unique(y)) == 1:
            return Node(value=np.bincount(y).argmax())
        
        feature, threshold = find_best_split(X, y)
        if feature is None:
            return Node(value=np.bincount(y).argmax())
        
        X_left, X_right, y_left, y_right = split(X, y, feature, threshold)
        
        left = build_tree(X_left, y_left, depth + 1)
        right = build_tree(X_right, y_right, depth + 1)
        
        return Node(feature, threshold, left, right)
    
    root = build_tree(X, y)
    return root

def random_forest(X, y, n_trees=10, max_depth=5):
    """
    Thực hiện random forest từ đầu
    """
    def bootstrap_sample(X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    trees = []
    for _ in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X, y)
        tree = decision_tree(X_sample, y_sample, max_depth)
        trees.append(tree)
    
    return trees

def k_nearest_neighbors(X, y, X_test, k=3):
    """
    Thực hiện KNN từ đầu
    """
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    predictions = []
    for x in X_test:
        distances = [euclidean_distance(x, x_train) for x_train in X]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y[k_indices]
        predictions.append(np.bincount(k_nearest_labels).argmax())
    
    return np.array(predictions)

def naive_bayes(X, y):
    """
    Thực hiện Naive Bayes từ đầu
    """
    class NaiveBayes:
        def __init__(self):
            self.priors = {}
            self.means = {}
            self.stds = {}
        
        def fit(self, X, y):
            classes = np.unique(y)
            for c in classes:
                X_c = X[y == c]
                self.priors[c] = len(X_c) / len(X)
                self.means[c] = np.mean(X_c, axis=0)
                self.stds[c] = np.std(X_c, axis=0)
        
        def predict(self, X):
            predictions = []
            for x in X:
                posteriors = []
                for c in self.priors:
                    prior = np.log(self.priors[c])
                    likelihood = np.sum(np.log(stats.norm.pdf(x, self.means[c], self.stds[c])))
                    posterior = prior + likelihood
                    posteriors.append(posterior)
                predictions.append(np.argmax(posteriors))
            return np.array(predictions)
    
    model = NaiveBayes()
    model.fit(X, y)
    return model

def cross_validation(X, y, model_type, k=5):
    """
    Thực hiện cross-validation từ đầu
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k
    scores = []
    
    for i in range(k):
        # Split data
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k-1 else n_samples
        
        X_test = X[start_idx:end_idx]
        y_test = y[start_idx:end_idx]
        X_train = np.vstack([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        # Train and evaluate
        if model_type == 'logistic':
            weights, bias = logistic_regression(X_train, y_train)
            predictions = sigmoid(np.dot(X_test, weights) + bias)
            score = np.mean((predictions > 0.5) == y_test)
        elif model_type == 'knn':
            predictions = k_nearest_neighbors(X_train, y_train, X_test)
            score = np.mean(predictions == y_test)
        elif model_type == 'naive_bayes':
            model = naive_bayes(X_train, y_train)
            predictions = model.predict(X_test)
            score = np.mean(predictions == y_test)
        
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

def feature_importance(X, y):
    """
    Tính feature importance từ đầu
    """
    def gini_importance(X, y, feature):
        X_left = X[X[:, feature] <= np.mean(X[:, feature])]
        X_right = X[X[:, feature] > np.mean(X[:, feature])]
        y_left = y[X[:, feature] <= np.mean(X[:, feature])]
        y_right = y[X[:, feature] > np.mean(X[:, feature])]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        gini_parent = 1 - np.sum((np.bincount(y) / len(y)) ** 2)
        gini_left = 1 - np.sum((np.bincount(y_left) / len(y_left)) ** 2)
        gini_right = 1 - np.sum((np.bincount(y_right) / len(y_right)) ** 2)
        
        return gini_parent - (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
    
    n_features = X.shape[1]
    importance = np.zeros(n_features)
    
    for feature in range(n_features):
        importance[feature] = gini_importance(X, y, feature)
    
    # Vẽ biểu đồ feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_features), importance)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance

if __name__ == "__main__":
    # Đọc dữ liệu mẫu
    df = pd.read_csv('iris.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Logistic Regression
    weights, bias = logistic_regression(X, y)
    
    # Decision Tree
    tree = decision_tree(X, y)
    
    # Random Forest
    forest = random_forest(X, y)
    
    # KNN
    X_test = X[:10]  # Lấy 10 mẫu đầu làm test
    X_train = X[10:]  # Phần còn lại làm train
    y_test = y[:10]
    y_train = y[10:]
    predictions = k_nearest_neighbors(X_train, y_train, X_test)
    
    # Naive Bayes
    nb_model = naive_bayes(X, y)
    
    # Cross Validation
    cv_mean, cv_std = cross_validation(X, y, 'logistic')
    
    # Feature Importance
    importance = feature_importance(X, y) 