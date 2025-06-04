from util import parse_numeric_condition_to_sql
from db import execute_sql_and_fetch
from datetime import datetime, timedelta
from util import extract_recent_days

def query_db_by_condition(question: str):
    sql = parse_numeric_condition_to_sql(question)
    if not sql:
        return []
    recent_days = extract_recent_days(question)
    if recent_days:
        cutoff = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        # SQL 문 끝에 날짜 조건 추가 (WHERE 절이 이미 있다고 가정)
        if "where" in sql.lower():
            sql += f" AND date >= '{cutoff}'"
        else:
            sql += f" WHERE date >= '{cutoff}'"

    return execute_sql_and_fetch(sql)
