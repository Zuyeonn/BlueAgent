import sqlite3
import json

conn = sqlite3.connect("user_data.db")

# SQL 명령 실행/결과 가져오기
cursor = conn.cursor()

# 테이블 생성
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    ppg_avg REAL,
    ppg_json TEXT,
    hrv REAL,
    stress REAL
);
""")

cursor.execute("DROP TABLE IF EXISTS user_data")

cursor.execute("""
CREATE TABLE user_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    ppg_json TEXT,
    hrv REAL,
    stress REAL
)
""")

data = [
    {"name": "김민지", "date": "2025-04-01", "hrv": 33, "stress": 90, "ppg_json": [0.91, 0.88, 1.02]},
    {"name": "김민지", "date": "2025-04-02", "hrv": 30, "stress": 91, "ppg_json": [0.95, 0.89, 1.01]},
    {"name": "김민지", "date": "2025-04-03", "hrv": 28, "stress": 94, "ppg_json": [0.96, 0.91, 1.00]},
    {"name": "김민지", "date": "2025-04-04", "hrv": 25, "stress": 95, "ppg_json": [0.97, 0.92, 1.03]},
    {"name": "김민지", "date": "2025-04-05", "hrv": 22, "stress": 97, "ppg_json": [1.00, 0.99, 1.01]},
    {"name": "김민지", "date": "2025-04-06", "hrv": 20, "stress": 98, "ppg_json": [1.02, 1.01, 0.98]},
    {"name": "김민지", "date": "2025-04-07", "hrv": 17, "stress": 99, "ppg_json": [1.03, 1.05, 1.00]},
    {"name": "이지훈", "date": "2025-04-01", "hrv": 55, "stress": 32, "ppg_json": [0.45, 0.52, 0.48]},
    {"name": "이지훈", "date": "2025-04-02", "hrv": 54, "stress": 33, "ppg_json": [0.50, 0.49, 0.53]},
]

conn = sqlite3.connect("user_data.db")
cursor = conn.cursor()

for row in data:
    ppg_json_str = json.dumps(row["ppg_json"])
    cursor.execute("""
        INSERT INTO user_data (name, date, ppg_json, hrv, stress)
        VALUES (?, ?, ?, ?, ?)
    """, (row["name"], row["date"], ppg_json_str, row["hrv"], row["stress"]))

conn.commit()
conn.close()

print("데이터 삽입 완료!")
