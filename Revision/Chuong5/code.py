import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def handle_missing_data(df):
    """
    Xử lý dữ liệu thiếu bằng các phương pháp thống kê cơ bản
    """
    # Phát hiện dữ liệu thiếu
    missing = df.isna().sum()
    print("Số lượng giá trị thiếu:")
    print(missing)
    
    # Xóa dữ liệu thiếu
    df_cleaned = df.dropna()
    
    # Điền giá trị thiếu
    df_filled = df.copy()
    
    # Xử lý riêng cho từng loại cột
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(exclude=['int64', 'float64']).columns
    
    # Điền giá trị thiếu cho cột số bằng trung bình
    for column in numeric_columns:
        mean_value = df[column].mean()
        df_filled[column] = df[column].fillna(mean_value)
    
    # Điền giá trị thiếu cho cột phân loại bằng giá trị phổ biến nhất
    for column in categorical_columns:
        mode_value = df[column].mode()[0]
        df_filled[column] = df[column].fillna(mode_value)
    
    return df_cleaned, df_filled

def handle_outliers(df):
    """
    Xử lý outliers sử dụng Z-score và IQR
    """
    # Chỉ xử lý outliers cho các cột số
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[numeric_columns]
    
    # Tính Z-score
    z_scores = np.abs((df_numeric - df_numeric.mean()) / df_numeric.std())
    
    # Phát hiện outliers dựa trên Z-score
    outliers_zscore = (z_scores > 3).sum()
    print("Số lượng outliers (Z-score):")
    print(outliers_zscore)
    
    # Xóa outliers dựa trên Z-score
    df_cleaned_zscore = df.copy()
    df_cleaned_zscore[numeric_columns] = df_numeric[(z_scores <= 3).all(axis=1)]
    
    # Phát hiện outliers dựa trên IQR
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    print("Số lượng outliers (IQR):")
    print(outliers_iqr)
    
    # Xóa outliers dựa trên IQR
    df_cleaned_iqr = df.copy()
    df_cleaned_iqr[numeric_columns] = df_numeric[~((df_numeric < (Q1 - 1.5 * IQR)) | 
                                                  (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Thay thế outliers bằng giá trị trung bình
    df_replaced = df.copy()
    for column in numeric_columns:
        mean_value = df[column].mean()
        df_replaced[column] = df[column].mask((df[column] < (Q1[column] - 1.5 * IQR[column])) | 
                                             (df[column] > (Q3[column] + 1.5 * IQR[column])), 
                                             mean_value)
    
    return df_cleaned_zscore, df_cleaned_iqr, df_replaced

def normalize_data(df):
    """
    Chuẩn hóa dữ liệu sử dụng Z-score, Min-Max và Robust scaling
    """
    # Chỉ chuẩn hóa các cột số
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[numeric_columns]
    
    # Chuẩn hóa Z-score
    z_score_scaled = df.copy()
    z_score_scaled[numeric_columns] = (df_numeric - df_numeric.mean()) / df_numeric.std()
    
    # Chuẩn hóa Min-Max
    min_max_scaled = df.copy()
    min_max_scaled[numeric_columns] = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
    
    # Chuẩn hóa Robust (sử dụng IQR)
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    robust_scaled = df.copy()
    robust_scaled[numeric_columns] = (df_numeric - df_numeric.median()) / IQR
    
    return z_score_scaled, min_max_scaled, robust_scaled

def transform_data(df):
    """
    Biến đổi dữ liệu sử dụng log, Box-Cox và đa thức
    """
    # Chỉ biến đổi các cột số
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df[numeric_columns]
    
    # Biến đổi log
    log_transformed = df.copy()
    log_transformed[numeric_columns] = np.log1p(df_numeric)
    
    # Biến đổi Box-Cox
    def box_cox_transform(x, lambda_param):
        if lambda_param == 0:
            return np.log(x)
        return (x ** lambda_param - 1) / lambda_param
    
    # Tìm lambda tối ưu cho Box-Cox
    def find_optimal_lambda(x):
        lambdas = np.linspace(-2, 2, 100)
        best_lambda = 0
        best_skewness = float('inf')
        
        for lambda_param in lambdas:
            transformed = box_cox_transform(x, lambda_param)
            skewness = stats.skew(transformed)
            if abs(skewness) < best_skewness:
                best_skewness = abs(skewness)
                best_lambda = lambda_param
        
        return best_lambda
    
    # Áp dụng Box-Cox cho từng cột
    box_cox_transformed = df.copy()
    for column in numeric_columns:
        optimal_lambda = find_optimal_lambda(df[column])
        box_cox_transformed[column] = box_cox_transform(df[column], optimal_lambda)
    
    # Biến đổi đa thức bậc 2
    poly_transformed = df.copy()
    for column in numeric_columns:
        poly_transformed[f"{column}_squared"] = df[column] ** 2
    
    return log_transformed, box_cox_transformed, poly_transformed

def visualize_transformations(df, column):
    """
    Vẽ biểu đồ so sánh các phương pháp biến đổi
    """
    # Chuẩn hóa dữ liệu
    z_score_scaled, min_max_scaled, robust_scaled = normalize_data(df)
    
    # Biến đổi dữ liệu
    log_transformed, box_cox_transformed, poly_transformed = transform_data(df)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 10))
    
    # Dữ liệu gốc
    plt.subplot(3, 2, 1)
    plt.hist(df[column], bins=30)
    plt.title('Original Data')
    
    # Z-score scaling
    plt.subplot(3, 2, 2)
    plt.hist(z_score_scaled[column], bins=30)
    plt.title('Z-score Scaling')
    
    # Min-Max scaling
    plt.subplot(3, 2, 3)
    plt.hist(min_max_scaled[column], bins=30)
    plt.title('Min-Max Scaling')
    
    # Robust scaling
    plt.subplot(3, 2, 4)
    plt.hist(robust_scaled[column], bins=30)
    plt.title('Robust Scaling')
    
    # Log transformation
    plt.subplot(3, 2, 5)
    plt.hist(log_transformed[column], bins=30)
    plt.title('Log Transformation')
    
    # Box-Cox transformation
    plt.subplot(3, 2, 6)
    plt.hist(box_cox_transformed[column], bins=30)
    plt.title('Box-Cox Transformation')
    
    plt.tight_layout()
    plt.savefig('transformation_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Đọc dữ liệu mẫu
    df = pd.read_csv('iris.csv')
    
    # Xử lý dữ liệu thiếu
    df_cleaned, df_filled = handle_missing_data(df)
    
    # Xử lý outliers
    df_cleaned_zscore, df_cleaned_iqr, df_replaced = handle_outliers(df)
    
    # Chuẩn hóa dữ liệu
    z_score_scaled, min_max_scaled, robust_scaled = normalize_data(df)
    
    # Biến đổi dữ liệu
    log_transformed, box_cox_transformed, poly_transformed = transform_data(df)
    
    # Vẽ biểu đồ so sánh
    visualize_transformations(df, 'sepal_length') 