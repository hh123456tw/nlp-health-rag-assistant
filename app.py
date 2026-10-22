import json
import os

from flask import Flask, Response, render_template, request
from ollama import Client

from query import search

app = Flask(__name__)

# 設定 Ollama API 的 URL
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_HOST = "https://112c-136-66-88-68.ngrok-free.app"
# OLLAMA_HOST = "https://{NGROK_URL}" # 如果使用 ngrok，請取消註解並替換 {NGROK_URL} 為實際的 ngrok URL

# 設定模型名稱和歷史對話檔案
MODEL_NAME = "gemma4:e2b"  # gpt-oss:20b # qwen3.5:0.8b # gemma4:e2b
HISTORY_FILE = "chat_history.json"

# 初始化 Ollama Client
client = Client(host=OLLAMA_HOST, timeout=600)


def build_context(sources):
    text = ""
    for i, news in enumerate(sources, start=1):
        text += f"【新聞 {i}】標題：{news['title']}\n"
        text += f"摘要：{news['summary']}\n\n"
    return text


# 讀取歷史對話
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# 儲存歷史對話
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# 將我們自己的歷史格式轉成 Ollama 需要的格式
def build_ollama_messages(history):
    # 先建立一個空的 messages list
    messages = []

    # 轉換歷史對話成 Ollama messages
    for msg in history:
        role = msg["role"]  # "user" 或 "model"
        text = msg["text"]  # 對話內容

        # 前端沿用 model，但 Ollama 需要 assistant
        if role == "model":
            role = "assistant"

        # 將訊息加入 messages list
        messages.append({"role": role, "content": text})

    return messages


# 設定首頁路由
@app.route("/")
def index():
    return render_template("index.html")


# 設定 API 路由。用戶端可以透過這個 API 來取得歷史對話。
@app.route("/api/history")
def get_history():
    return load_history()


# 設定 API 路由。用戶端可以透過這個 API 來發送訊息給 Ollama，並串流回應。
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    # 取得使用者訊息
    data = request.get_json()

    # 取得使用者訊息
    user_message = data["message"]
    use_rag = data.get("use_rag", False)
    sources = []
    if use_rag:
        sources = search(user_message)
        print(sources)
    # 1. 讀取舊歷史訊息
    history = load_history()

    # 2. 把歷史轉成 Ollama messages
    messages = build_ollama_messages(history)

    prompt = user_message
    if sources:
        prompt = (
            "請只根據下列新聞回答。\n\n"
            + build_context(sources)
            + "問題："
            + user_message
        )

    # 3. 加入這次使用者的新問題
    messages.append({"role": "user", "content": prompt})

    # 4. 生成回應，並串流回傳給前端
    def generate():
        full_reply = ""

        # 透過 Ollama Client 串流回應
        stream = client.chat(
            model=MODEL_NAME, messages=messages, keep_alive="1h", stream=True
        )

        # 串流回應的每一個部分
        for part in stream:
            # 取得這個部分的文字
            text = part["message"]["content"]

            # 如果有文字，就回傳給前端
            if text:
                full_reply += text
                yield text
        if sources:
            yield "\n\n參考新聞：\n"
            for news in sources:
                yield f"- {news['title']}｜{news['link']}｜{news['date']}\n"

        # 儲存使用者訊息到歷史對話
        history.append({"role": "user", "text": user_message})

        # 儲存模型回應到歷史對話
        history.append({"role": "model", "text": full_reply})

        # 儲存歷史對話到檔案
        save_history(history)

    # 5. 回傳串流回應給前端
    return Response(generate(), mimetype="text/plain")


# 清除歷史對話
@app.route("/api/clear", methods=["POST"])
def clear_history():
    save_history([])
    return {"status": "ok"}


# 啟動 Flask 應用程式
if __name__ == "__main__":
    app.run(debug=True)
