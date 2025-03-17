import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def load_and_prepare_data():
    # Đọc dữ liệu
    df = pd.read_csv('iris.csv')
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = scaler.fit_transform(df[numeric_columns])
    
    return df, X, numeric_columns

def principal_component_analysis(X, numeric_columns):
    # Thực hiện PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Vẽ biểu đồ explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Số thành phần chính')
    plt.ylabel('Tỷ lệ phương sai giải thích tích lũy')
    plt.title('Biểu đồ phương sai giải thích tích lũy')
    plt.savefig('pca_variance_ratio.png')
    plt.close()
    
    # Vẽ biểu đồ scatter của 2 thành phần chính đầu tiên
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Biểu đồ scatter của 2 thành phần chính đầu tiên')
    plt.savefig('pca_scatter.png')
    plt.close()
    
    return X_pca, pca

def clustering_analysis(X):
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    # Vẽ biểu đồ so sánh các phương pháp clustering
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-means Clustering')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title('DBSCAN Clustering')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png')
    plt.close()
    
    return kmeans_labels, dbscan_labels

def time_series_analysis():
    # Tạo dữ liệu chuỗi thời gian mẫu
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = np.random.normal(0, 1, 100).cumsum()
    ts = pd.Series(data, index=dates)
    
    # Kiểm tra tính dừng
    adf_result = adfuller(ts)
    print("\nKiểm tra ADF:")
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    
    # Phân tích mùa vụ
    decomposition = seasonal_decompose(ts, period=30)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(ts)
    plt.title('Chuỗi thời gian gốc')
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal')
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residual')
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')
    plt.close()
    
    return ts, decomposition

def regression_analysis(df, numeric_columns):
    # Chọn biến phụ thuộc và độc lập
    X = df[numeric_columns[:-1]]
    y = df[numeric_columns[-1]]
    
    # Thêm hằng số cho mô hình
    X = sm.add_constant(X)
    
    # Fit mô hình hồi quy
    model = sm.OLS(y, X).fit()
    
    # In kết quả
    print("\nKết quả hồi quy:")
    print(model.summary())
    
    # Vẽ biểu đồ residual
    plt.figure(figsize=(10, 6))
    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Residual')
    plt.title('Biểu đồ residual')
    plt.savefig('regression_residuals.png')
    plt.close()
    
    return model

def hypothesis_testing(df, numeric_columns):
    # Kiểm định t cho hai mẫu
    group1 = df[numeric_columns[0]][:50]
    group2 = df[numeric_columns[0]][50:]
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print("\nKiểm định t:")
    print(f'T-statistic: {t_stat}')
    print(f'p-value: {p_value}')
    
    # Kiểm định chi-square
    contingency_table = pd.crosstab(df['species'], pd.qcut(df[numeric_columns[0]], q=3))
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("\nKiểm định chi-square:")
    print(f'Chi-square statistic: {chi2}')
    print(f'p-value: {p_value}')
    
    return t_stat, p_value, chi2

if __name__ == "__main__":
    print("=== Chương 6: Phân tích dữ liệu nâng cao ===")
    
    # Chuẩn bị dữ liệu
    df, X, numeric_columns = load_and_prepare_data()
    
    # 1. Phân tích thành phần chính (PCA)
    print("\n1. Phân tích thành phần chính (PCA):")
    X_pca, pca = principal_component_analysis(X, numeric_columns)
    print("Tỷ lệ phương sai giải thích:", pca.explained_variance_ratio_)
    
    # 2. Phân tích phân cụm
    print("\n2. Phân tích phân cụm:")
    kmeans_labels, dbscan_labels = clustering_analysis(X)
    
    # 3. Phân tích chuỗi thời gian
    print("\n3. Phân tích chuỗi thời gian:")
    ts, decomposition = time_series_analysis()
    
    # 4. Phân tích hồi quy
    print("\n4. Phân tích hồi quy:")
    model = regression_analysis(df, numeric_columns)
    
    # 5. Kiểm định giả thuyết
    print("\n5. Kiểm định giả thuyết:")
    t_stat, p_value, chi2 = hypothesis_testing(df, numeric_columns) 