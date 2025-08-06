def normalize_column_name(question: str):
    import re
    replacements = {
        r"(스트레스|스트레스\s*지수|피로도)": "stress",
        r"(ppg|맥파|광용적맥파|혈류)": "ppg",
        r"(hrv|심박\s*변이도|자율\s*신경)": "hrv",
    }
    matched = False
    for pattern, replacement in replacements.items():
        new_question, count = re.subn(pattern, replacement, question, flags=re.IGNORECASE)
        if count > 0:
            question = new_question
            matched = True
    return question, matched

def parse_numeric_condition_to_sql(question: str):
    import re
    pattern = r"(stress|ppg|hrv)\s*(이|가|은|는)?\s*(\d+(\.\d+)?)\s*(이상|초과|이하|미만)"
    match = re.search(pattern, question)
    if match:
        column, _, value, _, cond = match.groups()
        op = {
            "이상": ">=", "초과": ">", "이하": "<=", "미만": "<"
        }.get(cond)
        return f"SELECT name, date, {column} AS value FROM user_data WHERE {column} {op} {value}"
    return None

def detect_unknown_keywords(question: str):
    unknown_keywords = ["맥박률", "심박수", "심박률", "맥박수", "산소포화도", "체온", "호흡수"]
    for kw in unknown_keywords:
        if kw in question:
            return f"'{kw}'은(는) 인식할 수 없는 항목입니다. 예: '스트레스', 'ppg', 'hrv' 등으로 다시 입력해보세요."
    return None

def extract_plot_target(question, allowed_columns=["ppg", "hrv", "stress"]):
    for col in allowed_columns:
        if col in question.lower():
            return col
    return "ppg"  # 기본값

def extract_python_code(text):
    import re
    # Response: 이후만 추출
    if "Response:" in text:
        text = text.split("Response:")[-1].strip()
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if "import matplotlib.pyplot as plt" in line:
            return "\n".join(lines[i:]).strip()
    return text.strip()

def extract_recent_days(question: str) -> int | None:
    import re
    # 숫자 기반 표현
    match = re.search(r"(최근\s*(\d+)\s*일|(\d+)\s*일\s*간)", question)
    if match:
        number = match.group(2) or match.group(3)
        return int(number)

    # 자연어 표현 매핑
    keyword_map = {
        "일주일": 7,
        "일 주일": 7,
        "한 주": 7,
        "1주일": 7,
        "한달": 30,
        "한 달": 30,
        "1개월": 30,
        "한달간": 30
    }

    for keyword, days in keyword_map.items():
        if keyword in question:
            return days

    return None  # 기간 표현이 없으면 None


def extract_date_or_month(question: str):
    import re
    from datetime import datetime
    # 일 단위 ("6월 15일", "7/2")
    pattern_day = r"(\d{1,2})[월/\s]*(\d{1,2})[일]?"
    match_day = re.search(pattern_day, question)
    if match_day:
        month, day = match_day.groups()
        try:
            year = datetime.today().year
            dt = datetime(year, int(month), int(day))
            return {"type": "day", "value": dt.strftime("%Y-%m-%d")}
        except:
            pass
    # 월 단위 ("6월", "2024년 6월", "6월달")
    pattern_month = r"(?:(\d{4})년\s*)?(\d{1,2})월(달)?"
    match_month = re.search(pattern_month, question)
    if match_month:
        year, month, _ = match_month.groups()
        year = int(year) if year else datetime.today().year
        try:
            dt = datetime(year, int(month), 1)
            return {"type": "month", "value": dt.strftime("%Y-%m")}
        except:
            pass
    return None
