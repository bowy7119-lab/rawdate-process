import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
import re

# --- 基本設定 ---
load_dotenv()
st.set_page_config(page_title="Egyptian Greek Inscription Analyzer", layout="wide")
CHROMA_PATH = "./chroma_db_store"
DATA_FILE = "egypt_data_enriched.json"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- データロード (キャッシュ) ---
@st.cache_resource
def get_chroma_db():
    if not os.path.exists(CHROMA_PATH): return None
    return chromadb.PersistentClient(path=CHROMA_PATH).get_collection("inscriptions")

@st.cache_data
def load_json_data():
    if not os.path.exists(DATA_FILE): return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# --- ① UIコンポーネント: 出典リスト表示 ---
def render_citation_list(inscriptions, max_items=20, title_prefix="ヒットした碑文"):
    """
    IDと年代をリスト表示し、クリックで原文（折り返し）と英訳を表示する共通関数
    """
    st.markdown(f"### 📜 {title_prefix} (Top {min(len(inscriptions), max_items)})")
    
    for item in inscriptions[:max_items]:
        # ヘッダー部分
        label = f"**ID: {item['id']}** | Date: {item.get('date_min')} ~ {item.get('date_max')} | {item.get('region_sub', 'Unknown')}"
        
        with st.expander(label):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Greek:**")
                # ギリシア語を折り返して表示するためにmarkdownを使用
                st.markdown(f"<div style='word-wrap: break-word;'>{item['text']}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("**English Translation:**")
                st.write(item.get('english_translation', '(No translation)'))

# --- ② ロジック: AIによるクエリ拡張 (修正版) ---
def get_smart_search_terms(user_query):
    """
    ユーザーの入力から、検索に必要な「英語概念」と「ギリシア語の全変化形」を生成する
    """
    system_prompt = """
    You are an expert Ancient Greek Historian.
    Analyze the user's query and return a JSON object with two lists.
    
    IMPORTANT: You must generate ACTUAL GREEK WORDS, not placeholders.
    
    Example Input: "καισαρ"
    Example Output:
    {
      "greek_forms": ["καισαρ", "καισαρος", "καισαρι", "καισαρα", "καισαρων", "καισαρσι"],
      "english_keywords": ["Caesar", "Emperor", "Imperial"]
    }

    Task:
    1. "greek_forms": List the lemma AND ALL inflected forms (nom/gen/dat/acc, sg/pl).
    2. "english_keywords": English translations and related concepts.
    """
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # 創造性を下げて確実に答えさせる
        )
        result = json.loads(res.choices[0].message.content)
        
        # 【保険】もしAIが入力した単語そのものを忘れていたら追加する
        if user_query not in result.get('greek_forms', []):
            result.setdefault('greek_forms', []).append(user_query)
            
        return result
    except:
        # エラー時は入力そのものを返す
        return {"greek_forms": [user_query], "english_keywords": [user_query]}

# --- ③ ロジック: 詳細検索 & 集計 ---
def analyze_data(data, search_terms):
    """
    全データを走査し、以下の3つを計算する
    1. 年代推移 (Line Chart用)
    2. 語形ごとのヒット数 (Pie Chart用)
    3. ヒットした碑文リスト
    """
    years_map = defaultdict(float)
    form_counts = defaultdict(int)
    matched_items = []
    
    # 検索語の準備
    greek_targets = [t for t in search_terms.get('greek_forms', []) if t]
    english_targets = [t.lower() for t in search_terms.get('english_keywords', []) if t]
    
    for d in data:
        is_hit = False
        text_greek = d['text'] # Case sensitive for Greek usually, but let's keep original
        text_eng = d.get('english_translation', '').lower()
        
        # A. ギリシア語形のマッチング（円グラフ用）
        # 正規表現を使わず、単純な包含確認を行う（高速化のため）
        for g_form in greek_targets:
            if g_form in text_greek:
                form_counts[g_form] += 1
                is_hit = True
        
        # B. 英語概念のマッチング（ヒット漏れ防止用）
        if not is_hit:
            for e_word in english_targets:
                if e_word in text_eng:
                    is_hit = True
                    break
        
        # ヒットした場合の年代集計
        if is_hit:
            matched_items.append(d)
            s, e = int(d.get('date_min', 0)), int(d.get('date_max', 0))
            if s == 0 and e == 0: continue
            
            duration = e - s + 1
            weight = 1.0 / duration if duration > 0 else 1.0
            for y in range(s, e + 1):
                years_map[y] += weight
                
    # データフレーム変換
    df_trend = pd.DataFrame(list(years_map.items()), columns=["Year", "Frequency"]).sort_values("Year")
    df_pie = pd.DataFrame(list(form_counts.items()), columns=["Form", "Count"]).sort_values("Count", ascending=False)
    
    return df_trend, df_pie, matched_items

# --- メイン UI ---
st.title("🏛️ Egyptian Greek Inscription Analyzer")
st.caption("Morphological Analysis & AI Historian")

collection = get_chroma_db()
full_data = load_json_data()

if collection is None:
    st.error("データベースが見つかりません。Step 2 を実行してください。")
    st.stop()

tab_trend, tab_chat = st.tabs(["📊 年代推移・語形分析", "🤖 歴史家チャット"])

# === Tab 1: 年代推移 & 円グラフ ===
with tab_trend:
    st.subheader("概念・語形変化の分析")
    query = st.text_input("検索語（例: καισαρ, プトレマイオス1世）", "καισαρ")
    
    if st.button("分析実行"):
        with st.spinner("AIが語形変化を展開し、全碑文を解析中..."):
            # 1. AI展開
            expanded = get_smart_search_terms(query)
            
            # ユーザーへのフィードバック
            with st.expander("🔍 AIが生成した検索ターゲット (クリックして確認)"):
                st.write(f"**Greek Forms:** {', '.join(expanded.get('greek_forms', []))}")
                st.write(f"**English Keywords:** {', '.join(expanded.get('english_keywords', []))}")
            
            # 2. 集計実行
            df_trend, df_pie, hits = analyze_data(full_data, expanded)
            
            if not df_trend.empty:
                # 3. グラフ表示
                col_graph1, col_graph2 = st.columns([2, 1])
                
                with col_graph1:
                    st.markdown("#### 📈 年代推移 (Frequency)")
                    fig_line = px.line(df_trend, x="Year", y="Frequency", title=f"Trend: {query}")
                    st.plotly_chart(fig_line, use_container_width=True)
                
                with col_graph2:
                    st.markdown("#### 🍰 語形出現比率")
                    if not df_pie.empty:
                        fig_pie = px.pie(df_pie, values="Count", names="Form", title="Greek Forms Distribution")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("ギリシア語形の直接一致はありませんでした（英語概念のみヒット）")

                # 4. 共通リスト形式で出典表示
                render_citation_list(hits, title_prefix="分析対象となった碑文")
                
            else:
                st.warning("該当するデータが見つかりませんでした。")

# === Tab 2: AIチャット ===
with tab_chat:
    st.subheader("Evidence-Based Chat")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("質問を入力 (例: プトレマイオス1世の統治について)"):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("AIが関連語を推論し、文献を検索中..."):
            # 1. AI推論 (Step 1)
            plan = get_smart_search_terms(prompt)
            search_text = " ".join(plan.get('english_keywords', []) + plan.get('greek_forms', []))
            
            # 2. ベクトル検索 (Step 2)
            # AIが考えた「ソテル」「ベレニケ」などの関連語も含めて検索
            q_vec = client.embeddings.create(input=[search_text], model="text-embedding-3-small").data[0].embedding
            results = collection.query(query_embeddings=[q_vec], n_results=20)
            
            # 3. コンテキスト構築
            context_str = ""
            ref_data = []
            id_map = {str(d['id']): d for d in full_data}
            
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context_str += f"[ID: {meta['id']}] {doc[:600]}...\n\n"
                # 元データを取得して参照リスト用にする
                original = id_map.get(str(meta['id']))
                if original:
                    ref_data.append(original)
            
            # 4. 回答生成 (Step 3)
            system_msg = """
            あなたは古代エジプト・ギリシア碑文の専門家です。
            ユーザーの質問に対し、提供された【Context】を証拠として用いながら、
            日本語で、学術的かつ論理的に回答してください。
            回答の中で主張を行う際は、必ず [ID: xxxxx] の形式で出典を明記してください。
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {prompt}"}
                ]
            )
            ans = response.choices[0].message.content
            
        st.chat_message("assistant").write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        
        # 5. 共通リスト形式でエビデンス表示
        with st.expander("📚 AIの検索戦略 & 参照エビデンス"):
            st.info(f"**AIが検索した関連語:** {', '.join(plan.get('english_keywords', [])[:10])} ...")
            render_citation_list(ref_data, title_prefix="回答に使用した碑文")