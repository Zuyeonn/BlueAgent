# app.py
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3
from handlers import (
    handle_visual,
    handle_report,
    handle_filter_rag,
    handle_rag_query
)
from util import normalize_column_name, detect_unknown_keywords
from rag_utils import load_rag_index
from flask_cors import CORS

#app = Flask(__name__)
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


# 모델 로드
model_path = "models" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# RAG 모델 로드
embedder, faiss_index, corpus = load_rag_index()

# DB 연결
conn = sqlite3.connect("user_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT name FROM user_data")
candidate_names = [row[0] for row in cursor.fetchall()]
chat_history = []


def classify_question(question: str):
    import re
    
    # 의미 없는 짧은 인삿말 또는 무의미 패턴
    if re.match(r"^(안녕|ㅎ+|하+|헐|뭐야|ㅋㅋ+|ㅎㅎ+|테스트|hi|hello)$", question.strip(), re.IGNORECASE):
        return "chitchat"
    # db에 있는 이름만 입력된 경우
    if any(name == question.strip() for name in candidate_names):
        return "name_only" 
    
    if re.search(r"(그래프|추이|그려줘|시계열|선 그래프)", question):
        return "visual"
    elif re.search(r"(평균|최대|최소|중앙값|요약|통계)", question):
        return "report"
    elif re.search(r"(이상|이하|보다 큰|보다 낮은|높은 사람|낮은 사람|조건에 맞는)", question):
        return "filter_rag"
    else:
        return "rag"
    

def extract_name_from_question(question: str, candidate_names: list):
    for name in candidate_names:
        if name in question:
            return name
    return None


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("message")
    if not user_question:
        return jsonify({"response": "질문이 비어 있습니다."})

    unknown = detect_unknown_keywords(user_question)
    if unknown:
        return jsonify({"intent": "unknown", "response": unknown})

    user_question, _ = normalize_column_name(user_question)
    chat_history.append({"role": "user", "content": user_question})

    intent = classify_question(user_question)
    target_name = extract_name_from_question(user_question, candidate_names)
    
    
    # 이름 유효성 검사 (visual, report, rag에만 해당)
    if intent in ["visual", "report", "rag"] and not target_name:
        return jsonify({
            "intent": intent,
            "response": "질문하신 사용자 이름을 찾을 수 없습니다. 올바른 이름을 입력해주세요."
        })
        
    if intent == "visual" and target_name:
        try:
            code = handle_visual(user_question, model, tokenizer, target_name, cursor)
            return jsonify({
                "intent": "visual",
                "response": "그래프가 생성되었습니다.",
                "image_base64": code
            })
        except Exception as e:
            return jsonify({
                "intent": "visual",
                "response": f"그래프 생성 중 오류가 발생했습니다: {str(e)}"
            })

    elif intent == "report":
        try:
            response = handle_report(user_question, model, tokenizer, candidate_names, cursor)
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"intent": "report", "response": response})
        except Exception as e:
            return jsonify({"intent": "report", "response": f"데이터 분석 중 오류가 발생했습니다: {str(e)}"})

    elif intent == "filter_rag":
        try:
            response = handle_filter_rag(user_question, model, tokenizer, chat_history)
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"intent": "filter_rag", "response": response})
        except Exception as e:
            return jsonify({"intent": "filter_rag", "response": f"데이터 필터링 중 오류가 발생했습니다: {str(e)}"})
        
    elif intent == "chitchat":
        return jsonify({
            "intent": "chitchat",
            "response": "안녕하세요! PPG, HRV, 스트레스 등에 대해 물어보시면 도와드릴 수 있어요 :)"
        })
        
    elif intent == "name_only":
        return jsonify({
            "intent": "name_only",
            "response": f"{target_name}님의 어떤 정보를 원하시나요? 예: 'ppg 평균 알려줘', '그래프 보여줘'"
        })

    else:
        try:
            response = handle_rag_query(user_question, model, tokenizer, embedder, faiss_index, corpus, candidate_names)
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"intent": "rag", "response": response})
        except Exception as e:
            return jsonify({"intent": "rag", "response": f"응답 생성 중 오류가 발생했습니다: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
