import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    # Đọc dữ liệu
    df = pd.read_csv('iris.csv')
    
    # Chuẩn bị features và target
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Mã hóa target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Chia tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X, y

def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test):
    # Danh sách các mô hình phân loại
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier()
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        # Huấn luyện mô hình
        clf.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = clf.predict(X_test)
        
        # Tính các metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
        
        # Vẽ confusion matrix cho mô hình tốt nhất
        if name == 'Random Forest':
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Random Forest')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig('confusion_matrix.png')
            plt.close()
    
    return results

def train_and_evaluate_regressors(X, y):
    # Chia dữ liệu cho hồi quy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Danh sách các mô hình hồi quy
    regressors = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'KNN': KNeighborsRegressor(),
        'SVM': SVR(),
        'XGBoost': xgb.XGBRegressor()
    }
    
    results = {}
    
    for name, reg in regressors.items():
        # Huấn luyện mô hình
        reg.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = reg.predict(X_test)
        
        # Tính các metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'R2': r2
        }
        
        # Vẽ biểu đồ so sánh giá trị thực và dự đoán cho mô hình tốt nhất
        if name == 'Random Forest':
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values - Random Forest')
            plt.savefig('regression_comparison.png')
            plt.close()
    
    return results

def cross_validation_analysis(X, y):
    # Danh sách các mô hình
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier()
    }
    
    cv_results = {}
    
    for name, model in models.items():
        # Thực hiện cross-validation
        scores = cross_val_score(model, X, y, cv=5)
        
        cv_results[name] = {
            'Mean CV Score': scores.mean(),
            'Std CV Score': scores.std()
        }
    
    return cv_results

def feature_importance_analysis(X, y):
    # Huấn luyện Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)
    
    # Lấy feature importance
    importance = rf.feature_importances_
    
    # Vẽ biểu đồ feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance

if __name__ == "__main__":
    print("=== Chương 7: Học máy cơ bản ===")
    
    # Chuẩn bị dữ liệu
    X_train, X_test, y_train, y_test, X, y = load_and_prepare_data()
    
    # 1. Huấn luyện và đánh giá các mô hình phân loại
    print("\n1. Kết quả phân loại:")
    classification_results = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test)
    for model, metrics in classification_results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # 2. Huấn luyện và đánh giá các mô hình hồi quy
    print("\n2. Kết quả hồi quy:")
    regression_results = train_and_evaluate_regressors(X, y)
    for model, metrics in regression_results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # 3. Phân tích cross-validation
    print("\n3. Kết quả cross-validation:")
    cv_results = cross_validation_analysis(X, y)
    for model, scores in cv_results.items():
        print(f"\n{model}:")
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
    
    # 4. Phân tích feature importance
    print("\n4. Feature Importance:")
    importance = feature_importance_analysis(X, y)
    for i, imp in enumerate(importance):
        print(f"Feature {i}: {imp:.4f}") 