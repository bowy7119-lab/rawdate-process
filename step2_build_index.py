import json
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 入力と出力の設定
INPUT_FILE = "egypt_data_enriched.json"
CHROMA_PATH = "./chroma_db_store"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ エラー: {INPUT_FILE} が見つかりません。Step 1 の完了を待ってください。")
        return

    print(f"📦 翻訳済みデータを読み込んでいます...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🏗️ ベクトルデータベースを構築します: {CHROMA_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 既存のコレクションをリセット（クリーンな構築のため）
    try:
        chroma_client.delete_collection("inscriptions")
    except:
        pass
    collection = chroma_client.create_collection("inscriptions")

    batch_size = 100
    print(f"🚀 検索用インデックスの作成を開始します（全{len(data)}件）...")
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        
        ids = [str(d['id']) for d in batch]
        documents = []
        metadatas = []
        
        for d in batch:
            # 検索精度向上のため、英訳を最優先に構成
            search_content = f"Translation: {d.get('english_translation', '')}\nGreek: {d.get('text', '')}"
            documents.append(search_content[:8000]) # トークン上限対策
            
            metadatas.append({
                "id": str(d['id']),
                "date_min": int(d.get('date_min', -9999)),
                "date_max": int(d.get('date_max', 9999)),
                "region": str(d.get('region_sub', 'Unknown'))
            })

        try:
            # ベクトル化（Embedding）
            response = client.embeddings.create(input=documents, model="text-embedding-3-small")
            embeddings = [item.embedding for item in response.data]
            
            # ChromaDBへ保存
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            continue

    print(f"\n✅ データベースの構築が完了しました！")

if __name__ == "__main__":
    main()