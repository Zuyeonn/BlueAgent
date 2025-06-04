import json
from generators import (
    generate_code_from_question,
    generate_report_from_question,
    generate_response_from_query_with_history,
    generate_rag_response
)
from util import extract_python_code
from util import extract_plot_target
from util import extract_recent_days
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

def handle_filter_rag(question, model, tokenizer, chat_history):
    from db_query import query_db_by_condition
    rows = query_db_by_condition(question)
    if not rows:
        return "조건에 맞는 결과가 없습니다."

    from generators import generate_response_from_query_with_history
    return generate_response_from_query_with_history(question, rows, chat_history, model, tokenizer)


def handle_rag_query(question, model, tokenizer, embedder, index, corpus, top_k=3):
    question_embedding = embedder.encode([question])
    # 유사 문서 검색
    D, I = index.search(question_embedding, top_k)

    # THRESHOLD = 1.0
    # if D[0][0] > THRESHOLD:
    #     return "이 질문은 관련 문서가 없어 정확한 답변이 어렵습니다."

    matched_docs = [corpus[i] for i in I[0]]
    return generate_rag_response(question, matched_docs, model, tokenizer)
