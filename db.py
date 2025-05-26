import sqlite3
import json

DB_PATH = "user_data.db"  # SQLite DB 파일 경로

def execute_sql_and_fetch(sql: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return rows

def fetch_and_compute_ppg_avg():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name, date, ppg_json FROM user_data")
    rows = cursor.fetchall()

    for name, date, ppg_json in rows:
        ppg_list = json.loads(ppg_json)
        avg = sum(ppg_list) / len(ppg_list)
        print(f"{name} ({date}) → PPG 평균: {avg:.2f}")

    conn.close()