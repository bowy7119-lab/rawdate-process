import json
import os
import stanza
from tqdm import tqdm

# 入出力ファイルの設定
INPUT_FILE = "egypt_data_enriched.json"
OUTPUT_FILE = "egypt_data_final.json"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ エラー: {INPUT_FILE} が見つかりません。Step 1 を完了してください。")
        return

    # 1. Stanzaの古代ギリシア語モデルをダウンロード＆初期化
    print("📥 Stanzaの古代ギリシア語モデルをダウンロードしています...")
    stanza.download('grc') # 初回のみダウンロードが走ります
    
    print("⚙️ 解析パイプラインを構築中...")
    # lemma: 辞書形化, pos: 品詞解析
    nlp = stanza.Pipeline('grc', processors='tokenize,lemma,pos', use_gpu=False)

    # 2. データの読み込み
    print(f"📦 データを読み込んでいます: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🚀 全{len(data)}件の形態素解析（レンマ化）を開始します...")
    
    # 3. 解析処理
    processed_count = 0
    for entry in tqdm(data, desc="解析中"):
        text = entry.get('text', '')
        if not text:
            entry['lemmas'] = []
            entry['tokens'] = []
            continue
            
        try:
            doc = nlp(text)
            
            # 検索・分析用にデータを構造化して保存
            # lemmas: 検索用の辞書形リスト (例: ['καισαρ', 'θεος'...])
            # tokens: 詳細分析用 (例: [{'word': 'θεου', 'lemma': 'θεος'}, ...])
            lemma_list = []
            token_details = []
            
            for sentence in doc.sentences:
                for word in sentence.words:
                    if word.lemma:
                        lemma_cleaned = word.lemma.lower()
                        lemma_list.append(lemma_cleaned)
                        token_details.append({
                            "word": word.text,       # 実際の語形 (例: θεου)
                            "lemma": lemma_cleaned,  # 辞書形 (例: θεος)
                            "pos": word.pos          # 品詞 (例: NOUN)
                        })
            
            entry['lemmas'] = list(set(lemma_list)) # 重複排除して検索高速化
            entry['analysis'] = token_details       # 円グラフ用に詳細を保存
            
        except Exception as e:
            # エラー時は空リストを入れてスキップ
            entry['lemmas'] = []
            entry['analysis'] = []
        
        processed_count += 1

    # 4. 保存
    print(f"💾 解析データを保存しています: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print("✅ 完了！ これで完璧な語形検索ができるようになりました。")
    print("次は app.py を更新して、この新しいデータを読み込ませましょう。")

if __name__ == "__main__":
    main()