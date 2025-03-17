import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import svd
from scipy.spatial.distance import cdist

def principal_component_analysis(X, n_components=2):
    """
    Thực hiện PCA từ đầu
    """
    # Chuẩn hóa dữ liệu
    X_centered = X - np.mean(X, axis=0)
    X_scaled = X_centered / np.std(X_centered, axis=0)
    
    # Tính ma trận hiệp phương sai
    cov_matrix = np.cov(X_scaled.T)
    
    # Tính eigenvalues và eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sắp xếp theo thứ tự giảm dần
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Chọn số thành phần
    eigenvectors = eigenvectors[:, :n_components]
    
    # Chiếu dữ liệu lên không gian mới
    X_pca = np.dot(X_scaled, eigenvectors)
    
    # Tính explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Vẽ biểu đồ explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             cumulative_variance_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Explained Variance Ratio')
    plt.savefig('pca_variance_ratio.png')
    plt.close()
    
    # Vẽ scatter plot của 2 PC đầu tiên
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA - First Two Components')
    plt.savefig('pca_scatter.png')
    plt.close()
    
    return X_pca, eigenvectors, explained_variance_ratio

def kmeans_clustering(X, n_clusters=3, max_iter=100):
    """
    Thực hiện K-means clustering từ đầu
    """
    # Khởi tạo centroids ngẫu nhiên
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
    
    for _ in range(max_iter):
        # Gán các điểm vào cluster gần nhất
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)
        
        # Cập nhật centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # Kiểm tra hội tụ
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids

def dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Thực hiện DBSCAN clustering từ đầu
    """
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1)  # -1 là noise
    cluster_id = 0
    
    def get_neighbors(point_idx):
        distances = cdist([X[point_idx]], X)[0]
        return np.where(distances <= eps)[0]
    
    def expand_cluster(point_idx, neighbors):
        nonlocal cluster_id
        labels[point_idx] = cluster_id
        
        for neighbor_idx in neighbors:
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                new_neighbors = get_neighbors(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    expand_cluster(neighbor_idx, new_neighbors)
    
    for point_idx in range(n_samples):
        if labels[point_idx] != -1:
            continue
            
        neighbors = get_neighbors(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1
            continue
            
        cluster_id += 1
        expand_cluster(point_idx, neighbors)
    
    return labels

def time_series_analysis(data, period=7):
    """
    Phân tích chuỗi thời gian từ đầu
    """
    n = len(data)
    
    # Tính trend bằng moving average
    trend = np.zeros_like(data)
    window_size = period
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        trend[i] = np.mean(data[start:end])
    
    # Tính seasonal component
    seasonal = np.zeros_like(data)
    for i in range(period):
        seasonal[i::period] = np.mean(data[i::period])
    
    # Tính residual
    residual = data - trend - seasonal
    
    # Vẽ kết quả
    plt.figure(figsize=(15, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(data)
    plt.title('Original Time Series')
    
    plt.subplot(4, 1, 2)
    plt.plot(trend)
    plt.title('Trend')
    
    plt.subplot(4, 1, 3)
    plt.plot(seasonal)
    plt.title('Seasonal')
    
    plt.subplot(4, 1, 4)
    plt.plot(residual)
    plt.title('Residual')
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')
    plt.close()
    
    return trend, seasonal, residual

def linear_regression(X, y):
    """
    Thực hiện hồi quy tuyến tính từ đầu
    """
    # Thêm cột bias
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Tính coefficients bằng phương pháp bình phương tối thiểu
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    # Tính dự đoán
    y_pred = X_b.dot(beta)
    
    # Tính R-squared
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Vẽ kết quả
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Results')
    plt.savefig('regression_results.png')
    plt.close()
    
    return beta, r2

if __name__ == "__main__":
    # Đọc dữ liệu mẫu
    df = pd.read_csv('iris.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # PCA
    X_pca, eigenvectors, explained_variance_ratio = principal_component_analysis(X)
    
    # K-means clustering
    kmeans_labels, centroids = kmeans_clustering(X)
    
    # DBSCAN clustering
    dbscan_labels = dbscan_clustering(X)
    
    # Time series analysis
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = np.random.normal(0, 1, 100).cumsum()
    trend, seasonal, residual = time_series_analysis(data)
    
    # Linear regression
    beta, r2 = linear_regression(X, y) 