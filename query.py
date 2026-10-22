"""
匯入套件
"""

import sqlite3

import faiss
from sentence_transformers import SentenceTransformer

# 基本設定
model_name = "BAAI/bge-m3"
encoder = SentenceTransformer(model_name)

# 讀取索引
index_path = "./vector.index"
index = faiss.read_index(index_path)


# 查詢句子
def search(question, k=3):
    list_query = [question]

    # 將查詢句子轉換成向量
    embeddings = encoder.encode(
        list_query, batch_size=3, show_progress_bar=False, normalize_embeddings=True
    )

    # 查詢
    D, I = index.search(embeddings, k=k)

    # 顯示結果
    list_scores = D.tolist()
    list_ids = I.tolist()

    # 透過 Document ID 查詢對應的新聞標題與內容
    conn = sqlite3.connect("./news.db")
    try:
        results = []
        # 透過 Document ID 查詢對應的新聞標題與內容
        for doc_id, score in zip(list_ids[0], list_scores[0]):

            # 透過 Document ID 查詢對應的新聞標題與內容
            row = conn.execute(
                "SELECT title, summary,link,date FROM news WHERE id = ?",
                (int(doc_id),),
            ).fetchone()

            # 取得新聞標題與內容，以及相似度分數
            title, summary, link, date = row

            results.append(
                {
                    "id": doc_id,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "date": date,
                    "score": score,
                }
            )
        return results
    finally:
        conn.close()


# 直接執行 python query.py 時才會跑這裡；被 app.py 匯入時不會執行
if __name__ == "__main__":
    print(search("有人用 AI 來偵測大腸瘜肉嗎？"))
