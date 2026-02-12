import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
import unicodedata
import re

# --- 基本設定 ---
load_dotenv()
st.set_page_config(page_title="Egyptian Greek Inscription Analyzer", layout="wide")

# パス設定
CHROMA_PATH = "./chroma_db_store"
DATA_FILE = "egypt_data_final.json" # データファイル（Step 1.5のものを使用しますが、lemmasがなくても動くように設計）

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- サイドバー設定 ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("AI Model")
    chat_model = st.selectbox(
        "Select Model",
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="gpt-4o: 高精度\ngpt-4o-mini: 高速"
    )
    st.divider()
    st.subheader("Chat History")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# --- データロード ---
@st.cache_resource
def get_chroma_db():
    if not os.path.exists(CHROMA_PATH): return None
    return chromadb.PersistentClient(path=CHROMA_PATH).get_collection("inscriptions")

@st.cache_data
def load_json_data():
    if not os.path.exists(DATA_FILE): return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# --- 🛠️ ヘルパー関数: 強力な正規化 ---
def normalize_text(text):
    """
    碑文検索用にテキストを正規化する。
    1. 小文字化
    2. アクセント・気息記号の除去 (NFD分解)
    3. 碑文記号 ([], (), <>, {}, .) の除去
    4. 異体字 (ς -> σ) の統一
    """
    if not text: return ""
    text = str(text).lower()
    
    # Unicode正規化 (アクセント分離して削除)
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    
    # 記号削除: [ ] ( ) < > { } .
    # これにより "[κ]αισαρ" -> "καισαρ" になる
    text = re.sub(r'[\[\]\(\)<>\{\}\.]', '', text)
    
    # ファイナルシグマ等の統一
    text = text.replace('ς', 'σ')
    
    # 余分な空白削除
    text = text.strip()
    
    return text

# --- 🧠 AIロジック: 検索語の拡張 ---
def get_expanded_search_terms(query):
    """
    ユーザーの入力から、検索すべき「ギリシア語の全変化形」と「英語キーワード」をAIにリストアップさせる
    """
    system_prompt = """
    You are an expert Ancient Greek Philologist.
    Analyze the user's query and return a JSON object with:
    1. "greek_forms": A list of the lemma AND ALL inflected forms (cases, numbers).
       Example: Input "καισαρ" -> Output ["καισαρ", "καισαρος", "καισαρι", "καισαρα", "καισαρων", "καισαρσι"]
       IMPORTANT: Normalize them (no accents).
    2. "english_keywords": English translations.
    """
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini", # 高速なモデルで十分
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(res.choices[0].message.content)
    except:
        # エラー時は入力そのまま返す
        return {"greek_forms": [query], "english_keywords": [query]}

# --- 📊 ロジック: データ分析 ---
def analyze_data_robust(data, query):
    years_map = defaultdict(float)
    form_counts = defaultdict(int)
    matched_items = []
    
    # 1. AIを使って検索語を拡張 (例: "καισαρ" -> ["καισαρ", "καισαρος", ...])
    expanded = get_expanded_search_terms(query)
    
    # 2. ターゲットを正規化セットに変換
    # AIが出した変化形をさらに正規化してセットにする
    target_greek = set([normalize_text(w) for w in expanded.get('greek_forms', [])])
    target_eng = set([w.lower() for w in expanded.get('english_keywords', [])])
    
    # 念のため、ユーザー入力そのものもターゲットに追加
    target_greek.add(normalize_text(query))

    for d in data:
        is_hit = False
        
        # --- A. ギリシア語検索 (正規化マッチング) ---
        text_raw = d.get('text', '')
        # 原文を単語分割
        # 記号込みでスペース区切りになっていることが多いので、まずはsplit
        words_raw = re.split(r'\s+', text_raw)
        
        for w_raw in words_raw:
            # 単語ごとに正規化 (例: "[κ]αισαρ" -> "καισαρ")
            w_norm = normalize_text(w_raw)
            
            # 正規化後の単語がターゲットに含まれるか？
            if w_norm in target_greek:
                is_hit = True
                if len(w_norm) > 1: # 1文字のゴミを除去
                    # 円グラフ用: 正規化後の形でカウント（表記ゆれを統一するため）
                    form_counts[w_norm] += 1
        
        # --- B. 英語検索 (救済措置) ---
        if not is_hit:
            content_eng = str(d.get('english_translation', '')).lower()
            # 英語キーワードのどれかが含まれているか
            for eng_key in target_eng:
                if eng_key in content_eng:
                    is_hit = True
                    break

        # --- 集計 ---
        if is_hit:
            s, e = int(d.get('date_min', 0)), int(d.get('date_max', 0))
            # 明らかにおかしい年代(0-0など)を除外するが、広範囲のものは許容
            if s == 0 and e == 0: 
                pass 
            else:
                duration = e - s + 1
                # 期間が長すぎるもの(500年以上など)はノイズになるので重みを下げる、あるいは除外も検討
                # ここでは単純に一様分布
                weight = 1.0 / duration if duration > 0 else 1.0
                for y in range(s, e + 1):
                    years_map[y] += weight
                matched_items.append(d)
            
    df_trend = pd.DataFrame(list(years_map.items()), columns=["Year", "Frequency"]).sort_values("Year")
    # 円グラフ: カウントが多い順
    df_pie = pd.DataFrame(list(form_counts.items()), columns=["Form", "Count"]).sort_values("Count", ascending=False)
    
    return df_trend, df_pie, matched_items, list(target_greek)

# --- UIコンポーネント: 出典リスト ---
def render_citation_list(inscriptions, max_items=20, title_prefix="ヒットした碑文"):
    st.markdown(f"### 📜 {title_prefix} (Top {min(len(inscriptions), max_items)})")
    
    seen_ids = set()
    unique_items = []
    for item in inscriptions:
        if item['id'] not in seen_ids:
            unique_items.append(item)
            seen_ids.add(item['id'])
            
    for item in unique_items[:max_items]:
        label = f"**ID: {item['id']}** | {item.get('date_min')}~{item.get('date_max')} | {item.get('region_sub', 'Unknown')}"
        with st.expander(label):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Greek:**")
                st.markdown(f"<div style='word-wrap: break-word; font-family: sans-serif; color: #aaa;'>{item['text']}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("**English Translation:**")
                st.write(item.get('english_translation', '(No translation)'))

# --- メイン UI ---
st.title("🏛️ Egyptian Greek Inscription Analyzer")
st.caption(f"Powered by AI & Robust Normalization | Model: {chat_model}")

collection = get_chroma_db()
full_data = load_json_data()

if collection is None or not full_data:
    st.error("データ準備が完了していません。Step 1 (または1.5), Step 2 を実行してください。")
    st.stop()

tab_trend, tab_chat = st.tabs(["📊 厳密語形分析", "🤖 歴史家チャット"])

# === Tab 1 ===
with tab_trend:
    st.subheader("AI推論と正規化による年代推移")
    query = st.text_input("検索語（例: καισαρ, ptolemy）", "καισαρ")
    
    if st.button("分析実行"):
        if not query:
            st.warning("検索語を入力してください")
        else:
            with st.spinner("AIが変化形を展開し、全データを照合中..."):
                df_trend, df_pie, hits, search_stems = analyze_data_robust(full_data, query)
                
                # 検索ターゲットの表示（最初の10個くらい）
                st.info(f"🔍 検索ターゲット(正規化済): {', '.join(list(search_stems)[:15])} ...")
                
                if not df_trend.empty:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"#### 📈 年代推移 (Hit: {len(hits)})")
                        fig_line = px.line(df_trend, x="Year", y="Frequency", title=f"Trend: {query}")
                        st.plotly_chart(fig_line, use_container_width=True)
                    with col2:
                        st.markdown("#### 🍰 語形出現比率 (正規化後)")
                        if not df_pie.empty:
                            fig_pie = px.pie(df_pie, values="Count", names="Form", title=f"Variations of '{query}'")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.caption("※ ギリシア語形の直接一致なし（英語概念ヒットのみ）")
                            
                    render_citation_list(hits, title_prefix="検索ヒット")
                else:
                    st.warning("該当データなし")

# === Tab 2 ===
with tab_chat:
    st.subheader("Evidence-Based Chat")
    
    if "history" not in st.session_state: st.session_state.history = []
    
    for m in st.session_state.history:
        st.chat_message(m["role"]).write(m["content"])
    
    if p := st.chat_input("質問を入力..."):
        st.session_state.history.append({"role": "user", "content": p})
        st.chat_message("user").write(p)
        
        with st.spinner(f"{chat_model} が検索中..."):
            # 1. 検索用クエリの生成 (AIで拡張)
            expanded = get_expanded_search_terms(p)
            search_text = " ".join(expanded.get('english_keywords', []) + expanded.get('greek_forms', []))
            
            # 2. ベクトル検索
            q_vec = client.embeddings.create(input=[search_text], model="text-embedding-3-small").data[0].embedding
            results = collection.query(query_embeddings=[q_vec], n_results=20)
            
            # 3. コンテキスト構築
            context_str = ""
            ref_data = []
            seen_refs = set()
            id_map = {str(d['id']): d for d in full_data}
            
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                mid = str(meta['id'])
                if mid not in seen_refs:
                    context_str += f"[ID: {mid}] {doc[:600]}...\n\n"
                    orig = id_map.get(mid)
                    if orig: ref_data.append(orig)
                    seen_refs.add(mid)
            
            # 4. 回答生成
            sys_msg = """
            あなたは古代エジプト・ギリシア碑文の専門家です。
            証拠【Context】に基づき、必ず [ID: xxxxx] を引用して日本語で回答してください。
            碑文特有の記号（[ ] や ( )）は、読みやすいように補完して解釈してください。
            """
            
            ans = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {p}"}
                ]
            ).choices[0].message.content
            
        st.chat_message("assistant").write(ans)
        st.session_state.history.append({"role": "assistant", "content": ans})
        
        with st.expander("📚 参照エビデンス"):
            render_citation_list(ref_data, title_prefix="参照データ")