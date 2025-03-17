import os
import csv
import json
import sqlite3
import re
from urllib.request import urlopen
from html.parser import HTMLParser
from datetime import datetime

class CSVHandler:
    """
    Xử lý file CSV từ đầu
    """
    @staticmethod
    def read_csv(file_path, delimiter=','):
        data = []
        headers = []
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            headers = next(csv_reader)
            for row in csv_reader:
                data.append(row)
        return headers, data
    
    @staticmethod
    def write_csv(data, headers, file_path, delimiter=','):
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file, delimiter=delimiter)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)

class JSONHandler:
    """
    Xử lý file JSON từ đầu
    """
    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    @staticmethod
    def write_json(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

class ExcelHandler:
    """
    Xử lý file Excel từ đầu (đơn giản hóa)
    """
    @staticmethod
    def read_excel(file_path):
        # Đọc file CSV với dấu tab làm delimiter
        headers, data = CSVHandler.read_csv(file_path, delimiter='\t')
        return headers, data
    
    @staticmethod
    def write_excel(data, headers, file_path):
        # Ghi file CSV với dấu tab làm delimiter
        CSVHandler.write_csv(data, headers, file_path, delimiter='\t')

class WebScraper(HTMLParser):
    """
    Web scraping từ đầu
    """
    def __init__(self):
        super().__init__()
        self.data = []
        self.current_tag = None
        self.current_attrs = None
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = attrs
    
    def handle_data(self, data):
        if self.current_tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if data.strip():
                self.data.append({
                    'tag': self.current_tag,
                    'text': data.strip(),
                    'attrs': self.current_attrs
                })
    
    def scrape_url(self, url):
        try:
            with urlopen(url) as response:
                html = response.read().decode('utf-8')
                self.feed(html)
                return self.data
        except Exception as e:
            print(f"Error scraping URL: {e}")
            return []

class DatabaseHandler:
    """
    Xử lý database từ đầu
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def create_table(self, table_name, columns):
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.cursor.execute(query)
        self.conn.commit()
    
    def insert_data(self, table_name, data):
        placeholders = ','.join(['?' for _ in data])
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        self.cursor.execute(query, data)
        self.conn.commit()
    
    def select_data(self, table_name, columns='*', where=None):
        query = f"SELECT {columns} FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def update_data(self, table_name, set_clause, where):
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where}"
        self.cursor.execute(query)
        self.conn.commit()
    
    def delete_data(self, table_name, where):
        query = f"DELETE FROM {table_name} WHERE {where}"
        self.cursor.execute(query)
        self.conn.commit()

class APIClient:
    """
    Tương tác với API từ đầu
    """
    def __init__(self, base_url):
        self.base_url = base_url
    
    def get(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        if params:
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            url += f"?{query_string}"
        
        try:
            with urlopen(url) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"Error making GET request: {e}")
            return None
    
    def post(self, endpoint, data):
        url = f"{self.base_url}/{endpoint}"
        try:
            data = json.dumps(data).encode('utf-8')
            with urlopen(url, data=data) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"Error making POST request: {e}")
            return None

def main():
    # Ví dụ sử dụng CSVHandler
    headers, data = CSVHandler.read_csv('../Chuong5/iris.csv')
    print("CSV Headers:", headers)
    print("First row:", data[0])
    
    # Ví dụ sử dụng JSONHandler
    sample_data = {
        'name': 'John',
        'age': 30,
        'city': 'New York'
    }
    JSONHandler.write_json(sample_data, 'sample.json')
    loaded_data = JSONHandler.read_json('sample.json')
    print("Loaded JSON:", loaded_data)
    
    # Ví dụ sử dụng WebScraper
    scraper = WebScraper()
    scraped_data = scraper.scrape_url('https://example.com')
    print("Scraped data:", scraped_data[:2])
    
    # Ví dụ sử dụng DatabaseHandler
    db = DatabaseHandler('test.db')
    db.connect()
    
    # Tạo bảng
    db.create_table('users', 'id INTEGER PRIMARY KEY, name TEXT, age INTEGER')
    
    # Thêm dữ liệu
    db.insert_data('users', (1, 'John', 30))
    
    # Truy vấn dữ liệu
    results = db.select_data('users')
    print("Database results:", results)
    
    db.disconnect()
    
    # Ví dụ sử dụng APIClient
    api = APIClient('https://api.example.com')
    response = api.get('users', {'page': 1})
    print("API response:", response)

if __name__ == "__main__":
    main() 