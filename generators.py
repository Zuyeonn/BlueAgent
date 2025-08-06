import torch
import re
import json
from util import extract_plot_target

def generate_code_from_question(question, model, tokenizer, rows, target_name):
    plot_target = extract_plot_target(question)

    if plot_target == "ppg":
        values = []
        for r in rows:
            if r[1] != target_name:
                continue
            ppg_raw = r[3]
            try:
                ppg_list = json.loads(ppg_raw) if isinstance(ppg_raw, str) else ppg_raw
            except json.JSONDecodeError:
                ppg_list = []
            if isinstance(ppg_list, list):
                values.append(sum(ppg_list) / len(ppg_list))
            elif isinstance(ppg_list, (int, float)):
                values.append(ppg_list)
    elif plot_target == "hrv":
        values = [r[4] for r in rows if r[1] == target_name]
    elif plot_target == "stress":
        values = [r[5] for r in rows if r[1] == target_name]
    else:
        values = []

    prompt = f"""
You are a Python programmer.

Task:
Write a Python script that uses matplotlib to plot the variable `values` over `dates`.

Constraints:
- Use the variables `dates` and `values` as already defined
- Do NOT define or assign them
- Import matplotlib.pyplot as plt
- Set figure size to 10x6
- Rotate x-axis labels by 45 degrees
- Use plt.tight_layout()
- Plot with plt.plot(dates, values)
- End with plt.show()

Output Format:
Output **only executable Python code**
Do NOT include markdown
Do NOT include explanations, comments, or examples
Start immediately with: import matplotlib.pyplot as plt

Response:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_report_from_question(question, model, tokenizer, summary_context):
    prompt = f"""당신은 사용자 건강 데이터를 간결하게 요약하는 시스템입니다.

아래는 유저들의 최근 건강 상태를 요약한 내용입니다:

{summary_context}

요약 예시:
- Response: 김유진의 HRV는 평균 25로 낮고, 스트레스는 95로 높은 편입니다.
- Response: 박민수는 PPG 변동성이 낮고, 스트레스 평균이 90 이상으로 안정적이지 않습니다.

질문: {question}

위 내용을 참고해 자연스럽고 간결한 한국어 문장으로 요약하세요.
**주의: 절대 '참고:'나 부가 설명은 쓰지 마세요.**
반드시 'Response:'로 시작하고, 구체적 수치는 문장 안에 녹여 쓰세요. 
Response: """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split("Response:")[-1].strip()


def generate_response_from_query_with_history(question, rows, chat_history, model, tokenizer):
    summary = "\n".join(f"{n}, {d}, {v}" for n, d, v in rows)
    history_context = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history[-6:]])
    prompt = f"""당신은 사용자 건강 조건을 판단하여 요약해주는 응답 시스템입니다.

아래는 직전 대화 내용입니다:
{history_context}

사용자의 질문:
{question}

조건을 만족하는 데이터 (이름, 날짜, 수치):
{summary}

예시 응답:
- Response: 김민지가 5월 3일, 5월 6일에 조건을 만족했습니다.
- Response: 박지훈은 6월 2일과 6월 5일에 스트레스 수치 95 이상을 보였습니다.

조건을 만족하는 사람과 날짜만 자연스럽게 요약하세요. 수치는 언급하지 마세요.  
반드시 'Response:'로 시작하는 한 문장으로만 답하세요.
Response: """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=400)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip().split("Response:")[-1].strip()

def generate_rag_response(question, context_docs, model, tokenizer):
    context = "\n".join(context_docs)
    prompt = f"""당신은 사용자 질문에 대해 배경 정보를 참고해 응답하는 시스템입니다.
사용자의 질문을 정확히 파악하고, 관련된 지표에 대해서만 답변하세요.

배경 문서:
{context}

질문:
{question}

요약 예시:
- Response: PPG는 광용적맥파를 의미하며, 스트레스 지수와 관련 있습니다.
- Response: HRV는 자율신경계의 균형을 판단하는 주요 지표입니다.

배경 문서를 참고하여 자연스럽고 간결한 한국어로 1~2문장으로 요약하세요.
위 문서가 질문과 관련이 없다면 "관련 정보를 찾을 수 없습니다"라고 답변하세요.
**주의: 절대 '참고:'나 부가 설명은 쓰지 마세요.**
반드시 'Response:'로 시작하세요.
Response: """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    return tokenizer.decode(output[0], skip_special_tokens=True).strip().split("Response:")[-1].strip()


# LLM 기반 intent fallback 분류
def generate_intent_from_llm(question: str, model, tokenizer) -> str:
    prompt = f"""
당신은 사용자 질문을 아래 목록 중 하나의 intent 유형으로 분류하는 시스템입니다.

각 intent의 정의는 다음과 같습니다:

- rag: 특정 수치(PPG, HRV, 스트레스 지수)의 의미, 기준, 정상 여부 등을 묻는 질문  
- report: 특정 사람의 지표에 대한 평균, 최대/최소값, 통계 등 요약을 요청하는 질문  
- visual: 특정 사람의 그래프나 시계열 시각화를 요청하는 질문  
- filter_rag: 
    1) 수치 조건(예: 90 이상, 100 미만 등)에 부합하는 사람을 찾는 질문  
    2) HRV, 스트레스, PPG 등을 기준으로 **안정적/불안정한 상태의 사람을 찾는 질문**
- stress_reason: 특정 사람의 스트레스가 높거나 낮은 이유를 묻는 질문  
- chitchat: 인사, 감탄, 테스트 등 대화의 시작이나 목적 없는 간단한 말  


사용자 질문:
\"{question}\"

해당 질문에 가장 적절한 의도 하나를 출력하세요. 반드시 '의도:'로 시작하세요.

의도:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=0.0)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"의도[:：]?\s*(rag|report|visual|filter_rag|stress_reason|chitchat)\b", decoded, re.IGNORECASE)

    return match.group(1).lower() if match else "ambiguous"


# 스트레스 원인 답변 프롬프트 (ppg, hrv 데이터 기반)
def generate_stress_reason_from_data(question, model, tokenizer, target_name, rows, specific_date=None):
    import json
    import numpy as np

    ppg_stds = []
    hrv_values = []

    for r in rows:
        if r[1] != target_name:
            continue
        try:
            ppg = json.loads(r[3]) if isinstance(r[3], str) else r[3]
        except:
            ppg = []
        if isinstance(ppg, list) and len(ppg) > 0:
            ppg_stds.append(np.std(ppg))
        hrv = r[4]
        if isinstance(hrv, (int, float)):
            hrv_values.append(hrv)

    if not hrv_values and not ppg_stds:
        return f"{target_name}님의 해당 날짜의 유효한 HRV/PPG 데이터가 부족합니다."

    # 수치 요약 텍스트
    summary_text = f"[{target_name}님의 {'해당 날짜' if specific_date else '최근 수치'} 요약]\n"
    if hrv_values:
        summary_text += f"- HRV 최근값: {hrv_values[-1]:.1f}, 평균: {np.mean(hrv_values):.2f}\n"
    if ppg_stds:
        summary_text += f"- PPG 변동성 최근값: {ppg_stds[-1]:.2f}, 평균: {np.mean(ppg_stds):.2f}\n"

    summary_context = summary_text.strip()

    prompt = f"""당신은 사용자 HRV, PPG 데이터를 기반으로 스트레스 원인을 설명하는 시스템입니다.

{summary_context}

질문: {question}

---

아래 조건을 바탕으로 스트레스 원인을 설명하는 1~2문장을 작성하세요:
- 최근값과 평균을 비교해, 수치가 얼마나 다른지 (높거나 낮음) 판단하세요.
- HRV가 평균보다 낮다면 스트레스가 증가할 수 있습니다.
- PPG 변동성이 높다면 스트레스 요인이 증가했을 수 있습니다.
- 두 지표가 모두 기준과 다르면 **두 가지 모두를 근거로** 설명하세요.
- 수치가 큰 차이가 없으면 “안정적”이나 “외부 요인 가능성”도 고려해 설명하세요.

---

예시 응답:
- Response: HRV가 평균보다 낮고, PPG 변동성이 높아 스트레스가 증가한 것으로 보입니다.
- Response: PPG 변동성이 증가했지만 HRV는 안정적입니다. 외부 요인 가능성이 있습니다.
- Response: HRV와 PPG 모두 평소 수준으로, 스트레스 증가 원인을 특정하기 어렵습니다.


구체적인 수치를 활용해 자연스럽고 간결한 한국어 문장으로 1~2문장 작성하세요.  
절대 '참고:'나 '요약:' 같은 말은 포함하지 마세요. 반드시 'Response:'로 시작하세요.

Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    reason = tokenizer.decode(output[0], skip_special_tokens=True).strip().split("Response:")[-1].strip()
    return summary_context + "\n\n[원인 분석]\n" + reason

