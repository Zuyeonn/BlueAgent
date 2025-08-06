import json
import statistics
from generators import (
    generate_code_from_question,
    generate_report_from_question,
    generate_response_from_query_with_history,
    generate_rag_response
)
from util import (extract_python_code, extract_plot_target, extract_recent_days, extract_date_or_month)
from datetime import datetime, timedelta

def handle_visual(question, model, tokenizer, target_name, cursor):
    import matplotlib
    matplotlib.use("Agg")
    import io
    import base64
    import matplotlib.pyplot as plt

    recent_days = extract_recent_days(question)
    if recent_days:
        cutoff = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        cursor.execute(
            "SELECT * FROM user_data WHERE name = ? AND date >= ? ORDER BY date",
            (target_name, cutoff)
        )
    else:
        cursor.execute(
            "SELECT * FROM user_data WHERE name = ? ORDER BY date",
            (target_name,)
        )

    rows = cursor.fetchall()
    if not rows:
        raise ValueError(f"{target_name}에 대한 데이터가 없습니다.")
    dates = [r[2] for r in rows]
    plot_target = extract_plot_target(question)
    print("시각화 대상:", plot_target)

    if plot_target == "ppg":
        # 평균 PPG 값으로 변환
        values = []
        for r in rows:
            raw = r[3]
            try:
                lst = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                lst = []
            if isinstance(lst, list):
                values.append(sum(lst)/len(lst))
            elif isinstance(lst, (int, float)):
                values.append(lst)
    elif plot_target == "hrv":
        values = [r[4] for r in rows]
    elif plot_target == "stress":
        values = [r[5] for r in rows]
    else:
        values = []

    llm_output = generate_code_from_question(question, model, tokenizer, rows, target_name)
    print("LLM output:\n", llm_output)
    code = extract_python_code(llm_output)

    plt.clf()
    plt.figure(figsize=(10, 6))
    exec(code, {"plt": plt, "dates": dates, "values": values})

    fig = plt.gcf()
    print("axes 개수:", len(fig.axes))
    if fig.axes:
        print("그래프 축 있음")
    else:
        print("plot 호출 안 됐거나 빈 데이터")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    print("base64 길이:", len(img_base64))

    return img_base64

def handle_report(question, model, tokenizer, candidate_names, cursor):
    target_name = next((name for name in candidate_names if name in question), None)
    if not target_name:
        return "특정 유저를 인식하지 못했습니다. 이름을 포함해서 다시 질문해주세요."

    recent_days = extract_recent_days(question)
    if recent_days:
        cutoff = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        cursor.execute("SELECT * FROM user_data WHERE name = ? AND date >= ?", (target_name, cutoff))
    else:
        cursor.execute("SELECT * FROM user_data WHERE name = ?", (target_name,))
    rows = cursor.fetchall()

    if not rows:
        return f"{target_name}의 해당 기간 데이터가 없습니다."

    ppglists = [json.loads(r[3]) for r in rows]
    ppg = [v for l in ppglists for v in l]
    hrv = [r[4] for r in rows]
    stress = [r[5] for r in rows]

    summary_context = f"- {target_name}:\n  PPG: {ppg}\n  HRV: {hrv}\n  스트레스: {stress}"
    response = generate_report_from_question(question, model, tokenizer, summary_context)
    return response.strip()


def is_stable(hrv_list, stress_list, ppg_json_list):
    if not hrv_list or not stress_list or not ppg_json_list:
        return False
    hrv_avg = sum(hrv_list) / len(hrv_list)
    stress_avg = sum(stress_list) / len(stress_list)
    ppg_flat = [v for sublist in ppg_json_list for v in sublist]
    ppg_avg = sum(ppg_flat) / len(ppg_flat)
    ppg_std = statistics.stdev(ppg_flat) if len(ppg_flat) > 1 else 0
    return (
        hrv_avg >= 50 and
        stress_avg <= 55 and
        0.95 <= ppg_avg <= 1.05 and
        ppg_std <= 0.08
    )
    
def is_unstable(hrv_list, stress_list, ppg_json_list):
    if not hrv_list or not stress_list or not ppg_json_list:
        return False
    hrv_avg = sum(hrv_list) / len(hrv_list)
    stress_avg = sum(stress_list) / len(stress_list)
    ppg_flat = [v for sublist in ppg_json_list for v in sublist]
    ppg_avg = sum(ppg_flat) / len(ppg_flat)
    ppg_std = statistics.stdev(ppg_flat) if len(ppg_flat) > 1 else 0
    signals = 0
    if hrv_avg <= 40:
        signals += 1
    if stress_avg >= 80:
        signals += 1
    if ppg_avg < 0.85 or ppg_avg > 1.15:
        signals += 1
    if ppg_std >= 0.1:
        signals += 1
    return signals >= 2


def handle_filter_rag(question, model, tokenizer, chat_history, cursor, user_names):
    from db_query import query_db_by_condition, execute_sql_and_fetch
    from util import parse_numeric_condition_to_sql, normalize_column_name
    from generators import generate_response_from_query_with_history
    import re
    from collections import defaultdict

    question, _ = normalize_column_name(question)
    
    # 안정성 여부 판별 질문이면 사용자별로 필터링
    unstable_mode = "불안정" in question
    stable_mode = "안정" in question and not unstable_mode

    if stable_mode or unstable_mode:
        recent_days = extract_recent_days(question)
        date_info = extract_date_or_month(question)
        stable_users = []
        unstable_users = []
        
        def has_batchim(word: str) -> bool:
            last_char = word[-1]
            code = ord(last_char)
            if 0xAC00 <= code <= 0xD7A3:
                return (code - 0xAC00) % 28 != 0
            return False

        for name in user_names:
            if recent_days:
                cutoff = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
                cursor.execute("SELECT ppg_json, hrv, stress FROM user_data WHERE name = ? AND date >= ?", (name, cutoff))
            elif date_info:
                if date_info["type"] == "day":
                    cursor.execute("SELECT ppg_json, hrv, stress FROM user_data WHERE name = ? AND date = ?", (name, date_info["value"]))
                elif date_info["type"] == "month":
                    cursor.execute("SELECT ppg_json, hrv, stress FROM user_data WHERE name = ? AND date LIKE ?", (name, date_info["value"] + "%"))
            else:
                cursor.execute("SELECT ppg_json, hrv, stress FROM user_data WHERE name = ?", (name,))

            rows = cursor.fetchall()
            hrv_list = [row[1] for row in rows if row[1] is not None]
            stress_list = [row[2] for row in rows if row[2] is not None]
            ppg_json_list = [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows if row[0]]

            if stable_mode and is_stable(hrv_list, stress_list, ppg_json_list):
                stable_users.append(name)
            elif unstable_mode and is_unstable(hrv_list, stress_list, ppg_json_list):
                unstable_users.append(name)

        if stable_mode:
            if stable_users:
                last = stable_users[-1]
                particle = "은" if has_batchim(last) else "는"
                return f"{', '.join(stable_users)}{particle} 안정적인 상태입니다."
            else:
                return "안정적인 상태로 분류된 사람이 없습니다."
        elif unstable_mode:
            if unstable_users:
                last = unstable_users[-1]
                particle = "은" if has_batchim(last) else "는"
                return f"{', '.join(unstable_users)}{particle} 불안정한 상태입니다."
            else:
                return "불안정한 상태로 분류된 사람이 없습니다."

    

    rows = query_db_by_condition(question)
    if rows:
        return generate_response_from_query_with_history(question, rows, chat_history, model, tokenizer)


    # SQL로 처리 불가능한 경우, 사용자별 평균 조건으로 간주
    column = None
    if "ppg" in question: column = "ppg_json"
    elif "hrv" in question: column = "hrv"
    elif "stress" in question: column = "stress"
    else:
        return "비교할 항목(ppg, hrv, stress)을 찾을 수 없습니다."

    match = re.search(r"(\d+(?:\.\d+)?)", question)
    if not match:
        return "비교 기준 숫자를 찾을 수 없습니다."
    threshold = float(match.group(1))

    if "이상" in question or "초과" in question:
        comparator = ">=" if "이상" in question else ">"
    elif "이하" in question or "미만" in question:
        comparator = "<=" if "이하" in question else "<"
    else:
        return "비교 연산자(이상, 이하 등)를 찾을 수 없습니다."

    # 전체 사용자 데이터 조회
    sql = f"SELECT name, {column} FROM user_data"
    rows = execute_sql_and_fetch(sql)
    user_data = defaultdict(list)
    for name, value in rows:
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, list):
                avg = sum(parsed) / len(parsed)
            else:
                avg = float(parsed)
        except:
            continue
        user_data[name].append(avg)

    # 평균 계산 + 필터링
    result = []
    for name, values in user_data.items():
        avg = sum(values) / len(values)
        if eval(f"{avg} {comparator} {threshold}"):
            result.append((name, avg))
    if not result:
        return "조건에 맞는 결과가 없습니다."

    # 직접 응답 생성 
    result.sort(key=lambda x: -x[1])  # 평균 높은 순
    response_lines = [f"- {name} (평균: {avg:.3f})" for name, avg in result]
    return "조건에 맞는 사람:\n" + "\n".join(response_lines)




def handle_rag_query(question, model, tokenizer, embedder, index, corpus, top_k=3):
    question_embedding = embedder.encode([question])
    # 유사 문서 검색
    D, I = index.search(question_embedding, top_k)

    # THRESHOLD = 1.0
    # if D[0][0] > THRESHOLD:
    #     return "이 질문은 관련 문서가 없어 정확한 답변이 어렵습니다."

    matched_docs = [corpus[i] for i in I[0]]
    return generate_rag_response(question, matched_docs, model, tokenizer)


# 스트레스 원인 답변 (ppg, hrv 데이터 기반) 
def handle_stress_reason(question, model, tokenizer, candidate_names, cursor):
    from util import normalize_column_name
    from generators import generate_stress_reason_from_data

    question, _ = normalize_column_name(question)
    target_name = next((name for name in candidate_names if name in question), None)
    if not target_name:
        return "누구의 스트레스 원인을 분석할지 이름을 입력해주세요."

    date_info = extract_date_or_month(question)
    recent_days = extract_recent_days(question) or 7

    # 1. 날짜 또는 월 단위
    if date_info:
        if date_info["type"] == "day":
            cursor.execute("SELECT * FROM user_data WHERE name = ? AND date = ?", (target_name, date_info["value"]))
        elif date_info["type"] == "month":
            start = date_info["value"] + "-01"
            end_dt = datetime.strptime(start, "%Y-%m-%d").replace(day=28) + timedelta(days=4)
            end = end_dt.replace(day=1).strftime("%Y-%m-%d")
            cursor.execute("SELECT * FROM user_data WHERE name = ? AND date >= ? AND date < ?", (target_name, start, end))
    else:
        cutoff = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        cursor.execute("SELECT * FROM user_data WHERE name = ? AND date >= ? ORDER BY date", (target_name, cutoff))

    rows = cursor.fetchall()
    if not rows:
        period = date_info["value"] if date_info else f"최근 {recent_days}일"
        return f"{target_name}님의 {period} 데이터가 없습니다."

    return generate_stress_reason_from_data(
        question=question,
        model=model,
        tokenizer=tokenizer,
        target_name=target_name,
        rows=rows,
        specific_date=date_info["value"] if date_info and date_info["type"] == "day" else None
    )

    matched_docs = [corpus[i] for i in I[0]]
    return generate_rag_response(question, matched_docs, model, tokenizer)
