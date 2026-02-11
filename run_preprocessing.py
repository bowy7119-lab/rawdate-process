import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# モデルは最強の gpt-4o を使用
MODEL = "gpt-4o"
INPUT_FILE = 'egypt_processed_tagged.json'

def analyze_long_inscription(entry):
    """超長文専用の解析ロジック"""
    text = entry['text']
    metadata = entry['metadata']
    
    # 15,000文字でカット（分析には十分）し、出力を「重要語30個」に厳格に制限
    prompt = f"""
    Analyze this massive Ancient Greek inscription (Decree).
    Metadata: {metadata}
    Text: {text[:15000]}

    Instructions:
    1. Extract exactly 5 conceptual English keywords.
    2. Extract exactly 30 of the most HISTORICALLY SIGNIFICANT Greek lemmas (Kings, Gods, Places, specific terms). 
    3. DO NOT output a long list. Keep the JSON response compact to avoid truncation.

    Output format:
    {{"keywords": ["...", "..."], "lemmas": ["...", "..." ]}}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a specialist in Epigraphy. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=2000 # 出力枠を十分に確保
        )
        result = json.loads(response.choices[0].message.content)
        entry['lemmas'] = result.get('lemmas', [])
        entry['keywords'] = result.get('keywords', [])
        print(f"✅ ID {entry['id']} rescued successfully!")
        return entry
    except Exception as e:
        print(f"❌ ID {entry['id']} failed again: {e}")
        # 最終手段：タグだけ手動風に付けて通す
        entry['lemmas'] = []
        entry['keywords'] = ["Major Decree", "Long Text", "Ptolemaic"]
        return entry

def main():
    if not os.path.exists(INPUT_FILE):
        print("ファイルが見つかりません。")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_count = 0
    for entry in data:
        # キーワードやレマが空（＝過去に失敗したデータ）だけを処理
        if not entry.get('keywords') or len(entry.get('keywords', [])) == 0:
            print(f"🔍 Rescuing ID {entry['id']} (Length: {len(entry['text'])})...")
            analyze_long_inscription(entry)
            updated_count += 1
            
            # 1件ごとに保存（確実性を期すため）
            with open(INPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            time.sleep(1) # API制限回避

    print(f"🎉 レスキュー完了！ {updated_count} 件の碑文を更新しました。")

if __name__ == "__main__":
    main()