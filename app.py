import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


DATA_PATH = "egypt_processed_tagged.json"
MODEL_NAME = "gpt-4o-mini"


def _get_api_key() -> str:
    load_dotenv()
    return os.getenv("OPENAI_API_KEY", "").strip()


@st.cache_data(show_spinner=False)
def load_data() -> List[Dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def generate_greek_keywords(user_query: str) -> Dict[str, List[str]]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if OpenAI is None:
        raise RuntimeError("openai package is not available")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a philologist. Given a short Japanese query about ancient inscriptions, "
        "return ONLY valid JSON with keys: expanded_greek, english_concepts. "
        "expanded_greek must include relevant Ancient Greek lemmas and major inflections. "
        "english_concepts should be short English concepts to broaden search."
    )
    user_prompt = f"Japanese query: {user_query}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"expanded_greek": [], "english_concepts": []}
    if "expanded_greek" not in data or "english_concepts" not in data:
        data = {"expanded_greek": data.get("expanded_greek", []), "english_concepts": data.get("english_concepts", [])}
    return data


def _build_regex(terms: List[str]) -> re.Pattern:
    escaped = [re.escape(t) for t in terms if t]
    if not escaped:
        return re.compile(r"a^")  # match nothing
    pattern = "(" + "|".join(escaped) + ")"
    return re.compile(pattern, flags=re.IGNORECASE)


def search_inscriptions(data: List[Dict], greek_terms: List[str]) -> List[Dict]:
    rx = _build_regex(greek_terms)
    results = []
    for item in data:
        text = str(item.get("text", ""))
        if rx.search(text):
            results.append(item)
    return results


def _year_range(date_min: int, date_max: int) -> Tuple[int, int]:
    if date_min is None or date_max is None:
        return None, None
    if date_min > date_max:
        date_min, date_max = date_max, date_min
    return date_min, date_max


def build_year_counts(items: List[Dict]) -> Dict[int, float]:
    counts: Dict[int, float] = defaultdict(float)
    for item in items:
        dmin = _safe_int(item.get("date_min"))
        dmax = _safe_int(item.get("date_max"))
        dmin, dmax = _year_range(dmin, dmax)
        if dmin is None or dmax is None:
            continue
        span = dmax - dmin + 1
        if span <= 0:
            continue
        weight = 1.0 / span
        for y in range(dmin, dmax + 1):
            counts[y] += weight
    return counts


@st.cache_data(show_spinner=False)
def build_total_year_counts(data: List[Dict]) -> Dict[int, float]:
    return build_year_counts(data)


def make_trend_df(match_counts: Dict[int, float], total_counts: Dict[int, float], normalize: bool):
    years = sorted(set(match_counts.keys()) | set(total_counts.keys()))
    rows = []
    for y in years:
        m = match_counts.get(y, 0.0)
        t = total_counts.get(y, 0.0)
        if normalize:
            value = m / t if t > 0 else 0.0
        else:
            value = m
        rows.append({"year": y, "value": value, "matched": m, "total": t})
    return rows


def build_context(items: List[Dict], max_items: int = 20) -> str:
    chunks = []
    for item in items[:max_items]:
        _id = item.get("id", "unknown")
        text = item.get("text", "")
        dmin = item.get("date_min", "")
        dmax = item.get("date_max", "")
        region = item.get("region", "")
        chunks.append(
            f"[ID: {_id}] Date: {dmin}–{dmax}; Region: {region}\nText: {text}"
        )
    return "\n\n".join(chunks)


def run_rag_chat(context: str, user_message: str) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a careful scholar. Answer ONLY from the provided inscriptions context. "
        "After each factual statement, cite evidence in the format [ID: 12345]. "
        "If the context is insufficient, say so explicitly and do not speculate."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def main():
    st.set_page_config(page_title="Egyptian Greek Inscription Analyzer", layout="wide")
    st.title("Egyptian Greek Inscription Analyzer")

    with st.status("Loading inscription data...", expanded=True) as status:
        data = load_data()
        status.update(label=f"Loaded {len(data):,} records.", state="complete")

    st.markdown("**Prompt**: 古代エジプト碑文高度分析アプリの構築")

    tab_trend, tab_chat = st.tabs(
        ["ギリシア語の年代推移", "碑文ベースのチャット"]
    )

    with tab_trend:
        st.header("ギリシア語の年代推移")
        st.caption("ギリシア語の語形を入力して、年代ごとの出現頻度を可視化します。")

        st.subheader("Query Expansion (任意)")
        user_query = st.text_input("日本語クエリ (例: 王, 神殿, 奉納)", "", key="jp_query")
        expand_clicked = st.button("ギリシア語クエリを生成", key="expand_btn")

        if "expanded" not in st.session_state:
            st.session_state.expanded = {"expanded_greek": [], "english_concepts": []}

        if expand_clicked and user_query:
            with st.spinner("OpenAIでクエリ拡張中..."):
                st.session_state.expanded = generate_greek_keywords(user_query)

        expanded = st.session_state.expanded
        st.write("**Expanded Greek:**", expanded.get("expanded_greek", []))
        st.write("**English Concepts:**", expanded.get("english_concepts", []))

        st.subheader("検索語 (ギリシア語)")
        manual_terms = st.text_input(
            "カンマ区切りで入力 (例: βασιλεύς, βασιλέως, βασιλεῖ)", ""
        )
        greek_terms = [t.strip() for t in manual_terms.split(",") if t.strip()]
        if not greek_terms:
            greek_terms = expanded.get("expanded_greek", [])

        normalize = st.checkbox("年代別総碑文数で正規化する", value=True, key="normalize_trend")
        search_clicked = st.button("検索を実行", key="search_btn")

        if search_clicked:
            if not greek_terms:
                st.warning("ギリシア語語形が未入力です。クエリ拡張または手入力してください。")
            else:
                with st.status("Searching inscriptions...", expanded=True) as status:
                    results = search_inscriptions(data, greek_terms)
                    status.update(label=f"Found {len(results):,} inscriptions.", state="complete")

                st.subheader("Trend Analysis (Uniform Distribution)")
                with st.spinner("Calculating trend..."):
                    match_counts = build_year_counts(results)
                    total_counts = build_total_year_counts(data)
                    rows = make_trend_df(match_counts, total_counts, normalize)
                fig = px.line(rows, x="year", y="value", title="Temporal Trend")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Hit List")
                for item in results:
                    _id = item.get("id", "unknown")
                    header = f"[ID: {_id}] {item.get('region', '')} ({item.get('date_min', '')}–{item.get('date_max', '')})"
                    with st.expander(header):
                        st.write(item.get("text", ""))
                        st.json(
                            {
                                "metadata": item.get("metadata"),
                                "keywords": item.get("keywords"),
                                "lemmas": item.get("lemmas"),
                            }
                        )

    with tab_chat:
        st.header("碑文ベースのチャット")
        st.caption("回答は必ず [ID: 12345] の形式でエビデンスを付与します。")

        st.subheader("検索語 (ギリシア語)")
        chat_terms = st.text_input(
            "カンマ区切りで入力 (例: βασιλεύς, βασιλέως, βασιλεῖ)", "",
            key="chat_terms"
        )
        chat_terms_list = [t.strip() for t in chat_terms.split(",") if t.strip()]
        chat_search_clicked = st.button("検索してチャットを開始", key="chat_search_btn")

        if "chat_results" not in st.session_state:
            st.session_state.chat_results = []

        if chat_search_clicked:
            if not chat_terms_list:
                st.warning("チャットのコンテキスト用にギリシア語語形を入力してください。")
            else:
                with st.status("Searching inscriptions for chat...", expanded=True) as status:
                    st.session_state.chat_results = search_inscriptions(data, chat_terms_list)
                    status.update(
                        label=f"Found {len(st.session_state.chat_results):,} inscriptions.",
                        state="complete",
                    )

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user_message = st.chat_input("碑文に関する質問を入力")
        if user_message:
            st.session_state.chat_messages.append({"role": "user", "content": user_message})
            with st.chat_message("user"):
                st.write(user_message)

            context = build_context(st.session_state.chat_results, max_items=25)
            with st.chat_message("assistant"):
                with st.spinner("回答生成中..."):
                    answer = run_rag_chat(context, user_message)
                    st.write(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
