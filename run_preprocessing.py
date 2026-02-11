import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルからAPIキーを読み込み
load_dotenv()

# --- 設定 ---
API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_FILE = 'egypt_dated_only.json'  # 年代特定済みのファイル
OUTPUT_FILE = 'egypt_processed_tagged.json' # 完成データ
MODEL = "gpt-4o" # 高速・安価・高性能

client = OpenAI(api_key=API_KEY)

def analyze_inscription(entry):
    """
    AI解析を試みるが、エラー（長文など）が発生した場合は
    解析をスキップして元のデータだけを保持して返す。
    """
    text = entry['text']
    
    # 1. そもそも極端に長いものはAIに投げずにスキップ（安全第一）
    if len(text) > 4000:
        print(f"⏩ ID {entry['id']} is very long. Skipping AI to ensure data safety.")
        entry['lemmas'] = [] # 空にしておく
        entry['keywords'] = ["Long Text", "Important Decree"] # 検索で見つかるよう最低限のタグ
        return entry

    # 2. 通常サイズのものだけAIに解析させる
    prompt = f"Analyze: {text[:2000]}\nOutput JSON: {{'lemmas':[], 'keywords':[]}}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=10 # 長引くようなら打ち切る
        )
        result = json.loads(response.choices[0].message.content)
        entry['lemmas'] = result.get('lemmas', [])
        entry['keywords'] = result.get('keywords', [])
        return entry
    except:
        # 3. エラーが出たら何もせず「生データ」として返す
        print(f"⚠️ ID {entry['id']} caused an error. Saving raw data only.")
        entry['lemmas'] = []
        entry['keywords'] = ["Unprocessed_Long_Text"]
        return entry
        
# --- メイン処理（並列実行） ---
def main():
    # データの読み込み
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # すでに処理済みのファイルがあれば読み込む（中断再開用）
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        processed_ids = {item['id'] for item in processed_data}
    else:
        processed_data = []
        processed_ids = set()

    # 未処理データの抽出
    to_process = [d for d in data if d['id'] not in processed_ids]
    print(f"全 {len(data)} 件中、未処理の {len(to_process)} 件を処理します...")

    # 並列処理（5スレッド程度が安全）
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(analyze_inscription, item): item for item in to_process}
        
        count = 0
        for future in futures:
            result = future.result()
            if result:
                processed_data.append(result)
                count += 1
                
                # 50件ごとに保存
                if count % 50 == 0:
                    print(f"{count} 件完了... 保存中")
                    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    # 最終保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print("全処理完了！ 'egypt_processed_tagged.json' を作成しました。")

if __name__ == "__main__":
    main()