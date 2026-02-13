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
DATA_FILE = "egypt_data_final.json" 

def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

api_key = get_openai_api_key()
if not api_key:
    st.error("OPENAI_API_KEY が未設定です。Streamlit secrets に設定してください。")
    st.stop()

client = OpenAI(api_key=api_key)

# --- サイドバー固定幅 & チャット履歴 ---
if "history" not in st.session_state:
    st.session_state.history = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

with st.sidebar:
    st.subheader("モード選択")
    tab_choice = st.radio(
        "",
        ["📊 年代推移", "💬 碑文チャット"],
        index=0 if st.session_state.get("active_tab") == "📊 年代推移" else 1,
    )
    st.session_state["active_tab"] = tab_choice

    st.divider()
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            width: 360px !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 360px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.analysis_history:
        st.subheader("分析履歴")
        for idx, item in enumerate(st.session_state.analysis_history[::-1]):
            title = item.get("title", f"Analysis {idx+1}")
            if st.button(f"📊 {title}", key=f"analysis_{idx}"):
                st.session_state["analysis_selected"] = item
                st.session_state["active_tab"] = "📊 年代推移"
                st.rerun()
    st.subheader("履歴")
    if st.session_state.conversations:
        for idx, conv in enumerate(st.session_state.conversations[::-1]):
            title = conv.get("title", f"Conversation {idx+1}")
            col_a, col_b = st.columns([5, 1])
            with col_a:
                if st.button(f"💬 {title}", key=f"conv_{idx}"):
                    st.session_state.history = conv.get("messages", [])
                    st.session_state.active_conversation = conv.get("id")
                    st.session_state["active_tab"] = "💬 碑文チャット"
                    st.rerun()
            with col_b:
                if st.button("🗑️", key=f"del_conv_{idx}"):
                    conv_id = conv.get("id")
                    st.session_state.conversations = [
                        c for c in st.session_state.conversations if c.get("id") != conv_id
                    ]
                    if st.session_state.active_conversation == conv_id:
                        st.session_state.active_conversation = None
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

# --- 🛠️ 共通ヘルパー: 強力な正規化 ---
def normalize_text(text):
    """
    碑文検索用にテキストを正規化する。
    アクセント除去、記号除去、異体字統一を行う。
    """
    if not text: return ""
    text = str(text).lower()
    
    # Unicode正規化 (アクセント分離して削除)
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    
    # 記号削除: [ ] ( ) < > { } .
    text = re.sub(r'[\[\]\(\)<>\{\}\.]', '', text)
    
    # ファイナルシグマ等の統一
    text = text.replace('ς', 'σ')
    
    return text.strip()

# --- 🧠 タブ1用ロジック: 単語分析用の拡張 ---
def get_expanded_search_terms(query):
    """(タブ1用) 単語レベルでの変化形展開"""
    system_prompt = """
    You are an expert Ancient Greek Philologist.
    Analyze the user's query and return a JSON object with:
    1. "greek_forms": A list of the lemma AND ALL inflected forms.
       Example: "καισαρ" -> ["καισαρ", "καισαρος", "καισαρι", "καισαρα", "καισαρων"]
       IMPORTANT: Normalize them (no accents).
    2. "english_keywords": English translations.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"greek_forms": [query], "english_keywords": [query]}

# --- 🧠 タブ2用ロジック: チャット用の高度な検索戦略 ---
def get_chat_search_strategy(user_question):
    """
    (タブ2用) ユーザーの質問(日本語可)から、歴史的背景を考慮した検索キーワードを生成する。
    例: "プトレマイオス1世" -> {"english": ["Ptolemy I", "Soter", "Berenice"], "greek": ["Πτολεμαῖος", "Σωτήρ", "Βερενίκη"]}
    """
    system_prompt = """
    You are an expert Historian of Ptolemaic and Roman Egypt.
    Analyze the user's question and extract key search terms to find relevant Greek inscriptions.
    
    Task:
    1. Identify key historical figures, deities, or concepts in the question.
    2. Expand them to include:
       - Specific epithets (e.g., "Ptolemy I" -> "Soter", "Lagi").
       - Associated family members (e.g., "Berenice").
       - Key Greek terms (e.g., "Basileus", "Synodos").
    3. Return a JSON object with:
       - "english": List of English keywords.
       - "greek": List of Ancient Greek keywords (lemmas or common forms).
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            response_format={"type": "json_object"},
            temperature=0.3 # 少し創造性を持たせて連想させる
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"english": [user_question], "greek": []}

# --- 📊 タブ1ロジック: データ分析 (前回と同じ) ---
def analyze_data_robust(data, query):
    years_map = defaultdict(float)
    form_counts = defaultdict(int)
    matched_items = []
    
    expanded = get_expanded_search_terms(query)
    target_greek = set([normalize_text(w) for w in expanded.get('greek_forms', [])])
    target_eng = set([w.lower() for w in expanded.get('english_keywords', [])])
    target_greek.add(normalize_text(query))

    for d in data:
        is_hit = False
        text_raw = d.get('text', '')
        # 記号を除去しつつ単語分割
        words_raw = re.split(r'\s+', text_raw)
        
        for w_raw in words_raw:
            w_norm = normalize_text(w_raw)
            if w_norm in target_greek:
                is_hit = True
                if len(w_norm) > 1:
                    form_counts[w_norm] += 1
        
        if not is_hit:
            content_eng = str(d.get('english_translation', '')).lower()
            for eng_key in target_eng:
                if eng_key in content_eng:
                    is_hit = True
                    break

        if is_hit:
            s, e = int(d.get('date_min', 0)), int(d.get('date_max', 0))
            if s == 0 and e == 0: pass 
            else:
                duration = e - s + 1
                weight = 1.0 / duration if duration > 0 else 1.0
                for y in range(s, e + 1):
                    years_map[y] += weight
                matched_items.append(d)
            
    df_trend = pd.DataFrame(list(years_map.items()), columns=["Year", "Frequency"]).sort_values("Year")
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
col_logo, col_title = st.columns([1, 12])
with col_logo:
    st.image("EGIAlogo.png", width=120)
with col_title:
    st.title("Egyptian Greek Inscription Analyzer")
st.caption("Powered by AI & Robust Normalization")

collection = get_chroma_db()
full_data = load_json_data()

if collection is None or not full_data:
    st.error("データ準備が完了していません。Step 1/1.5, Step 2 を実行してください。")
    st.stop()

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "📊 年代推移"

# === Tab 1: 年代推移 (完成済) ===
if tab_choice == "📊 年代推移":
    st.subheader("AI推論と正規化による年代推移")
    query = st.text_input("検索語（例: καισαρ, ptolemy）", "καισαρ")
    
    if st.button("分析実行"):
        if not query:
            st.warning("検索語を入力してください")
        else:
            with st.spinner("AIが変化形を展開し、全データを照合中..."):
                df_trend, df_pie, hits, search_stems = analyze_data_robust(full_data, query)
                
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
                    # Save analysis to sidebar history
                    title = f"{query} ({len(hits)} hits)"
                    st.session_state.analysis_history.append(
                        {
                            "title": title,
                            "query": query,
                            "hits": hits,
                            "trend": df_trend,
                            "pie": df_pie,
                            "search_stems": search_stems,
                        }
                    )
                else:
                    st.warning("該当データなし")

    # Show selected analysis from sidebar
    if st.session_state.get("analysis_selected"):
        sel = st.session_state["analysis_selected"]
        st.info(f"🔁 過去の分析: {sel.get('title')}")
        if sel.get("trend") is not None and not sel["trend"].empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"#### 📈 年代推移 (Hit: {len(sel.get('hits', []))})")
                fig_line = px.line(sel["trend"], x="Year", y="Frequency", title=f"Trend: {sel.get('query')}")
                st.plotly_chart(fig_line, use_container_width=True)
            with col2:
                st.markdown("#### 🍰 語形出現比率 (正規化後)")
                if sel.get("pie") is not None and not sel["pie"].empty:
                    fig_pie = px.pie(sel["pie"], values="Count", names="Form", title=f"Variations of '{sel.get('query')}'")
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.caption("※ ギリシア語形の直接一致なし（英語概念ヒットのみ）")
            render_citation_list(sel.get("hits", []), title_prefix="検索ヒット")

# === Tab 2: チャット機能 (アップデート版) ===
if tab_choice == "💬 碑文チャット":
    st.subheader("碑文チャット")
    if st.button("🆕 新しいチャット"):
        st.session_state.history = []
        st.session_state.active_conversation = None
    st.markdown("#### AI Model")
    chat_model = st.selectbox(
        "Select Model",
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="gpt-4o: 高精度\ngpt-4o-mini: 高速"
    )

    st.markdown(
        """
        <style>
        /* Chat input fixed at bottom */
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 0;
            left: 360px;
            right: 0;
            z-index: 1000;
            padding: 1rem 1rem 1.25rem;
            width: calc(100% - 360px);
            background: linear-gradient(180deg, rgba(14,14,18,0) 0%, rgba(14,14,18,0.85) 35%, rgba(14,14,18,1) 100%);
        }
        /* Shift input to align with main content width */
        @media (max-width: 1200px) {
            div[data-testid="stChatInput"] {
                left: 0;
                width: 100%;
            }
        }
        /* Make the input area larger, Gemini-like */
        div[data-testid="stChatInput"] textarea {
            min-height: 72px;
            font-size: 1rem;
            line-height: 1.4;
        }
        div[data-testid="stChatInput"] input {
            min-height: 56px;
            font-size: 1rem;
        }
        /* Keep input aligned with main content (avoid sidebar overlap) */
        div[data-testid="stChatInput"] > div {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }
        /* Align user messages to the right (Gemini-like) */
        div[data-testid="stChatMessage"][data-testid="stChatMessage-user"] {
            display: flex;
            justify-content: flex-end;
        }
        div[data-testid="stChatMessage"][data-testid="stChatMessage-user"] > div {
            display: flex;
            justify-content: flex-end;
        }
        div[data-testid="stChatMessage"][data-testid="stChatMessage-user"] [data-testid="stMarkdownContainer"] {
            display: inline-block;
            text-align: left;
            margin-left: auto;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.6rem 0.9rem;
            max-width: 75%;
        }
        /* Assistant messages remain left-aligned */
        div[data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] [data-testid="stMarkdownContainer"] {
            max-width: 80%;
        }
        /* Keep content above fixed input */
        .block-container {
            padding-bottom: 8.5rem;
        }
        /* Prevent unintended italics in model output */
        div[data-testid="stChatMessage"] em {
            font-style: normal !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    if "history" not in st.session_state: st.session_state.history = []
    
    for m in st.session_state.history:
        st.chat_message(m["role"]).write(m["content"])
        if m.get("role") == "assistant" and m.get("refs"):
            with st.expander("🔍 参照エビデンス"):
                render_citation_list(m["refs"], title_prefix="参照データ")
    
    if p := st.chat_input("質問を入力"):
        st.session_state.history.append({"role": "user", "content": p})
        st.chat_message("user").write(p)
        
        with st.spinner(f"{chat_model} が関連用語を推論し、検索中..."):
            
            # 1. 検索戦略の立案 (ここが進化)
            # 日本語の質問から、検索すべき英語・ギリシア語のキーワードを生成
            strategy = get_chat_search_strategy(p)
            
            # 検索用テキストを作成: 英語キーワード + ギリシア語キーワードを結合
            # ベクトル検索は「概念」を探すので、キーワードを羅列するのが効果的
            search_query = " ".join(strategy.get('english', []) + strategy.get('greek', []))
            
            # 2. ベクトル検索 (検索範囲を広めに30件)
            q_vec = client.embeddings.create(input=[search_query], model="text-embedding-3-small").data[0].embedding
            results = collection.query(query_embeddings=[q_vec], n_results=30)
            
            # 3. コンテキスト構築
            context_str = ""
            ref_data = []
            seen_refs = set()
            id_map = {str(d['id']): d for d in full_data}
            
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                mid = str(meta['id'])
                if mid not in seen_refs:
                    orig = id_map.get(mid)
                    date_min = orig.get("date_min") if orig else ""
                    date_max = orig.get("date_max") if orig else ""
                    region = orig.get("region_sub") if orig else ""
                    context_str += f"[ID: {mid}] Date: {date_min}–{date_max}; Region: {region}\n{doc[:600]}...\n\n"
                    if orig:
                        ref_data.append(orig)
                    seen_refs.add(mid)
            
            # 4. 回答生成
            sys_msg = """
            あなたは古代エジプト・ギリシア碑文の専門家です。
            提供された【Context】(英訳付き碑文)を根拠として用い、
            不足する歴史的背景は一般知識として補いながら、質問に対して日本語で詳細に回答してください。
            
            ルール:
            1. 碑文から根拠を引く場合は、必ず [ID: xxxxx] の形式で出典を明記してください。
            1. 可能な限り多くの該当碑文を引用し、代表的なものは複数挙げてください（少なくとも6件以上を目標）。
            2. 引用する碑文については必ず年代（date_min〜date_max）を明示し、その年代背景を加味して解説してください。
            3. 各碑文の解説には「年代背景・地域・事象（宗教/政治/社会）」のいずれかを含め、年代に即した分析を必ず行ってください。
            4. 歴史的背景・一般知識で補足する部分は文頭に「背景知識:」と明示し、出典IDは付けないでください。
            5. 文脈から、碑文の記述が質問に関連する理由を補足してください。
            6. 碑文中の記号（[ ]など）は、読みやすいように補って解釈してください。
            7. 回答は十分に長く、詳細にしてください。
            """
            
            try:
                ans_res = client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {p}"}
                    ]
                )
                ans = (ans_res.choices[0].message.content or "").strip()
            except Exception as e:
                ans = f"回答の生成に失敗しました。再度お試しください。\n\n詳細: {e}"
            if not ans:
                ans = "回答が空でした。再度お試しください。"
            
        st.chat_message("assistant").write(ans)
        st.session_state.history.append(
            {"role": "assistant", "content": ans, "refs": ref_data}
        )

        # Save conversation summary into sidebar list
        summary_title = p[:24] + ("…" if len(p) > 24 else "")
        if st.session_state.active_conversation is None:
            conv_id = f"conv_{len(st.session_state.conversations)+1}"
            st.session_state.conversations.append(
                {
                    "id": conv_id,
                    "title": summary_title,
                    "messages": st.session_state.history.copy(),
                    "refs": ref_data,
                }
            )
            st.session_state.active_conversation = conv_id
        else:
            for conv in st.session_state.conversations:
                if conv.get("id") == st.session_state.active_conversation:
                    conv["messages"] = st.session_state.history.copy()
                    conv["refs"] = ref_data
                    if conv.get("title", "") == "":
                        conv["title"] = summary_title
                    break
        
        # ユーザーに「どんな言葉で検索したか」を見せる（透明性）
        with st.expander("🔍 AIの検索戦略 & 参照エビデンス"):
            st.info(f"**AIが生成した検索語:**\n- English: {', '.join(strategy.get('english', []))}\n- Greek: {', '.join(strategy.get('greek', []))}")
            render_citation_list(ref_data, title_prefix="参照データ")
