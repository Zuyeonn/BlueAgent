import torch
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
        output = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip().split("Response:")[-1].strip()

def generate_rag_response(question, context_docs, model, tokenizer):
    context = "\n".join(context_docs)
    prompt = f"""당신은 사용자 질문에 대해 배경 정보를 참고해 응답하는 시스템입니다.

배경 문서:
{context}

질문:
{question}

요약 예시:
- Response: PPG는 광용적맥파를 의미하며, 스트레스 지수와 관련 있습니다.
- Response: HRV는 자율신경계의 균형을 판단하는 주요 지표입니다.

배경 문서를 참고하여 자연스럽고 간결한 한국어로 한 문장으로 요약하세요.
반드시 'Response:'로 시작하세요.
Response: """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    return tokenizer.decode(output[0], skip_special_tokens=True).strip().split("Response:")[-1].strip()
