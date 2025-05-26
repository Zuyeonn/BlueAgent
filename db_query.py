from util import parse_numeric_condition_to_sql
from db import execute_sql_and_fetch

def query_db_by_condition(question: str):
    sql = parse_numeric_condition_to_sql(question)
    if sql:
        return execute_sql_and_fetch(sql)
    return []
