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
