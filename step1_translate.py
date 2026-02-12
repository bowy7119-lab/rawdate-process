import json
import os
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "egypt_processed_tagged.json"
OUTPUT_FILE = "egypt_data_enriched.json"

def translate_entry(entry):
    # すでに翻訳済みならスキップ
    if "english_translation" in entry and entry["english_translation"]:
        return entry
    
    # 長すぎるテキストはカットして翻訳（コスト・エラー対策）
    text_snippet = entry['text'][:3000]
    
    try:
        # 簡易的かつ高速な翻訳
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the Ancient Greek inscription to English. Output ONLY the translation."},
                {"role": "user", "content": text_snippet}
            ],
            temperature=0
        )
        entry["english_translation"] = response.choices[0].message.content
    except Exception:
        entry["english_translation"] = ""
    
    return entry

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"エラー: {INPUT_FILE} が見つかりません。")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"🚀 {len(data)}件のデータの英訳処理を開始します...")
    
    # 10並列で高速処理
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(translate_entry, data), total=len(data)))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 英訳完了！ '{OUTPUT_FILE}' を作成しました。")

if __name__ == "__main__":
    main()