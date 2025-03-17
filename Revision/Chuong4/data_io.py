import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
from bs4 import BeautifulSoup
import sqlite3
import json

# 1. Đọc và ghi dữ liệu từ file
def read_write_files():
    # Đọc dữ liệu từ CSV
    df = pd.read_csv('../Chuong5/iris.csv')
    
    # Phân tích dữ liệu
    print("Thông tin cơ bản về dữ liệu:")
    print(df.info())
    print("\nThống kê mô tả:")
    print(df.describe())
    
    # Vẽ biểu đồ phân phối
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='sepal_length', bins=30)
    plt.title('Phân phối độ dài đài hoa')
    plt.savefig('sepal_length_distribution.png')
    plt.close()
    
    # Tính hệ số tương quan
    correlation = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Ma trận tương quan')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Tính khoảng tin cậy
    confidence_interval = stats.t.interval(alpha=0.95, 
                                         df=len(df)-1,
                                         loc=np.mean(df['sepal_length']),
                                         scale=stats.sem(df['sepal_length']))
    print(f"\nKhoảng tin cậy 95% cho độ dài đài hoa: {confidence_interval}")
    
    # Ghi dữ liệu đã xử lý
    df.to_csv('processed_iris.csv', index=False)
    
    # Ghi dữ liệu dạng JSON
    df.to_json('iris.json', orient='records')

# 2. Thu thập dữ liệu từ web
def web_scraping():
    # Ví dụ: Lấy dữ liệu từ một trang web
    url = "https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Lấy bảng dữ liệu
    tables = soup.find_all('table')
    if tables:
        df = pd.read_html(str(tables[0]))[0]
        print("\nDữ liệu từ web:")
        print(df.head())
        df.to_csv('largest_companies.csv', index=False)

# 3. Tương tác với API
def api_interaction():
    # Ví dụ: Lấy dữ liệu từ một API công khai
    url = "https://api.publicapis.org/entries"
    response = requests.get(url)
    data = response.json()
    
    # Chuyển đổi thành DataFrame
    df = pd.DataFrame(data['entries'])
    print("\nDữ liệu từ API:")
    print(df.head())
    df.to_csv('api_data.csv', index=False)

# 4. Làm việc với cơ sở dữ liệu
def database_operations():
    # Tạo kết nối SQLite
    conn = sqlite3.connect('example.db')
    
    # Đọc dữ liệu từ CSV vào SQLite
    df = pd.read_csv('../Chuong5/iris.csv')
    df.to_sql('iris', conn, if_exists='replace', index=False)
    
    # Thực hiện truy vấn
    query = "SELECT * FROM iris WHERE sepal_length > 5.0"
    result = pd.read_sql_query(query, conn)
    print("\nDữ liệu từ SQLite:")
    print(result.head())
    
    # Đóng kết nối
    conn.close()

if __name__ == "__main__":
    print("=== Chương 4: Đọc và ghi dữ liệu ===")
    
    print("\n1. Đọc và ghi dữ liệu từ file:")
    read_write_files()
    
    print("\n2. Thu thập dữ liệu từ web:")
    web_scraping()
    
    print("\n3. Tương tác với API:")
    api_interaction()
    
    print("\n4. Làm việc với cơ sở dữ liệu:")
    database_operations() 