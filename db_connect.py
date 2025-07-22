import sqlite3
import json
import random
from datetime import datetime, timedelta

# 상태별 범위 정의
states = {
    "calm": ([0.95, 1.00, 1.02], (60, 80), (20, 40)),
    "surprised": ([1.05, 1.08, 1.02], (20, 40), (70, 90)),
    "excited": ([1.10, 1.15, 1.05], (15, 30), (80, 100)),
    "normal": ([0.98, 1.01, 1.00], (45, 60), (40, 60)),
    "drowsy": ([0.85, 0.88, 0.91], (65, 85), (10, 30)),
    "anxious": ([1.00, 1.07, 1.04], (25, 45), (70, 85))
}

# 샘플 데이터 생성 함수
def generate_sample_data(state: str):
    ppg_base, hrv_range, stress_range = states[state]
    ppg = [round(v + random.uniform(-0.02, 0.02), 3) for v in ppg_base]
    hrv = random.randint(*hrv_range)
    stress = random.randint(*stress_range)
    return ppg, hrv, stress

# 이름 목록 (데이터 추가)
names = [
    "박지혜", "이하늘", "성지민", "김한나", "강민지", "박채영", "안지영", "정혜리", "이희선", "박지윤",
    "김수연", "한수연", "김승호", "조유리", "김예린", "이민정"
]


# 날짜 목록
start_date = datetime.strptime("2025-06-01", "%Y-%m-%d")
end_date = datetime.strptime("2025-07-15", "%Y-%m-%d")
all_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]

# DB 연결
conn = sqlite3.connect("/home/jhyoo_2/juyeon/blue_agent/user_data.db")  # 절대경로
cursor = conn.cursor()

# 테이블이 없으면 생성 (기존 데이터 보존)
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    ppg_json TEXT,
    hrv REAL,
    stress REAL
);
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

for row in data:
    cursor.execute("SELECT COUNT(*) FROM user_data WHERE name=? AND date=?", (row["name"], row["date"]))
    if cursor.fetchone()[0] == 0:
        ppg_json_str = json.dumps(row["ppg_json"])
        cursor.execute("""
            INSERT INTO user_data (name, date, ppg_json, hrv, stress)
            VALUES (?, ?, ?, ?, ?)
        """, (row["name"], row["date"], ppg_json_str, row["hrv"], row["stress"]))

# 자동 생성 샘플 데이터 
for name in names:
    selected_dates = random.sample(all_dates, 15)
    for date in selected_dates:
        cursor.execute("SELECT COUNT(*) FROM user_data WHERE name=? AND date=?", (name, date))
        if cursor.fetchone()[0] == 0:
            state = random.choice(list(states.keys()))
            ppg, hrv, stress = generate_sample_data(state)
            ppg_json = json.dumps(ppg)
            cursor.execute("""
                INSERT INTO user_data (name, date, ppg_json, hrv, stress)
                VALUES (?, ?, ?, ?, ?)
            """, (name, date, ppg_json, hrv, stress))

# 저장 후 종료
conn.commit()
conn.close()

print("데이터 삽입 완료") 
