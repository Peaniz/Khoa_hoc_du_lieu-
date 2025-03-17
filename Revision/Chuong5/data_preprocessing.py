import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import seaborn as sns

def load_and_analyze_data():
    # Đọc dữ liệu
    df = pd.read_csv('iris.csv')
    
    # Phân tích dữ liệu gốc
    print("Thống kê mô tả dữ liệu gốc:")
    print(df.describe())
    
    # Vẽ biểu đồ phân phối dữ liệu gốc
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(df.select_dtypes(include=[np.number]).columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=column, bins=30)
        plt.title(f'Phân phối {column}')
    plt.tight_layout()
    plt.savefig('original_distributions.png')
    plt.close()
    
    return df

def z_score_normalization(df):
    # Chuẩn hóa Z-score
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_zscore = pd.DataFrame(
        scaler.fit_transform(df[numeric_columns]),
        columns=numeric_columns
    )
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df_zscore, x=column, bins=30)
        plt.title(f'Z-score {column}')
    plt.tight_layout()
    plt.savefig('zscore_distributions.png')
    plt.close()
    
    return df_zscore

def min_max_scaling(df):
    # Chuẩn hóa Min-Max
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_minmax = pd.DataFrame(
        scaler.fit_transform(df[numeric_columns]),
        columns=numeric_columns
    )
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df_minmax, x=column, bins=30)
        plt.title(f'Min-Max {column}')
    plt.tight_layout()
    plt.savefig('minmax_distributions.png')
    plt.close()
    
    return df_minmax

def robust_scaling(df):
    # Chuẩn hóa Robust
    scaler = RobustScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_robust = pd.DataFrame(
        scaler.fit_transform(df[numeric_columns]),
        columns=numeric_columns
    )
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df_robust, x=column, bins=30)
        plt.title(f'Robust {column}')
    plt.tight_layout()
    plt.savefig('robust_distributions.png')
    plt.close()
    
    return df_robust

def log_transformation(df):
    # Biến đổi log
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_log = df[numeric_columns].apply(np.log1p)
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df_log, x=column, bins=30)
        plt.title(f'Log {column}')
    plt.tight_layout()
    plt.savefig('log_distributions.png')
    plt.close()
    
    return df_log

def box_cox_transformation(df):
    # Biến đổi Box-Cox
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_boxcox = df[numeric_columns].copy()
    
    for column in numeric_columns:
        df_boxcox[column], _ = stats.boxcox(df[column])
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 5))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df_boxcox, x=column, bins=30)
        plt.title(f'Box-Cox {column}')
    plt.tight_layout()
    plt.savefig('boxcox_distributions.png')
    plt.close()
    
    return df_boxcox

def compare_transformations(df):
    # So sánh các phương pháp chuẩn hóa
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    column = numeric_columns[0]  # Chọn một cột để so sánh
    
    plt.figure(figsize=(15, 10))
    
    # Dữ liệu gốc
    plt.subplot(2, 3, 1)
    sns.histplot(data=df, x=column, bins=30)
    plt.title('Dữ liệu gốc')
    
    # Z-score
    plt.subplot(2, 3, 2)
    sns.histplot(data=z_score_normalization(df), x=column, bins=30)
    plt.title('Z-score')
    
    # Min-Max
    plt.subplot(2, 3, 3)
    sns.histplot(data=min_max_scaling(df), x=column, bins=30)
    plt.title('Min-Max')
    
    # Robust
    plt.subplot(2, 3, 4)
    sns.histplot(data=robust_scaling(df), x=column, bins=30)
    plt.title('Robust')
    
    # Log
    plt.subplot(2, 3, 5)
    sns.histplot(data=log_transformation(df), x=column, bins=30)
    plt.title('Log')
    
    # Box-Cox
    plt.subplot(2, 3, 6)
    sns.histplot(data=box_cox_transformation(df), x=column, bins=30)
    plt.title('Box-Cox')
    
    plt.tight_layout()
    plt.savefig('comparison_transformations.png')
    plt.close()

if __name__ == "__main__":
    print("=== Chương 5: Xử lý dữ liệu ===")
    
    # Đọc và phân tích dữ liệu
    df = load_and_analyze_data()
    
    # Áp dụng các phương pháp chuẩn hóa
    print("\n1. Chuẩn hóa Z-score:")
    df_zscore = z_score_normalization(df)
    print(df_zscore.describe())
    
    print("\n2. Chuẩn hóa Min-Max:")
    df_minmax = min_max_scaling(df)
    print(df_minmax.describe())
    
    print("\n3. Chuẩn hóa Robust:")
    df_robust = robust_scaling(df)
    print(df_robust.describe())
    
    print("\n4. Biến đổi Log:")
    df_log = log_transformation(df)
    print(df_log.describe())
    
    print("\n5. Biến đổi Box-Cox:")
    df_boxcox = box_cox_transformation(df)
    print(df_boxcox.describe())
    
    # So sánh các phương pháp
    print("\nSo sánh các phương pháp chuẩn hóa:")
    compare_transformations(df) 