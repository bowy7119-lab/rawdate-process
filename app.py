import json
import os
import re
from collections import defaultdict
from math import sqrt
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
EMBED_MODEL = "text-embedding-3-small"
EMBED_CACHE_PATH = ".cache/embeddings.json"


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


def translate_to_japanese(text: str) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a careful translator of Ancient Greek inscriptions. "
        "Translate the provided inscription into concise, accurate Japanese. "
        "Do not add interpretations not grounded in the text."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _embed_texts(texts: List[str]) -> List[List[float]]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if OpenAI is None:
        raise RuntimeError("openai package is not available")

    client = OpenAI(api_key=api_key)
    embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([e.embedding for e in resp.data])
    return embeddings


def _build_embedding_text(item: Dict) -> str:
    text = str(item.get("text", ""))
    lemmas = item.get("lemmas", [])
    keywords = item.get("keywords", [])
    parts = [text]
    if isinstance(lemmas, list):
        parts.append(" ".join(str(x) for x in lemmas))
    else:
        parts.append(str(lemmas))
    if isinstance(keywords, list):
        parts.append(" ".join(str(x) for x in keywords))
    else:
        parts.append(str(keywords))
    return "\n".join(p for p in parts if p)


@st.cache_data(show_spinner=False)
def load_or_create_embeddings(data: List[Dict]) -> List[List[float]]:
    if os.path.exists(EMBED_CACHE_PATH):
        with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("count") == len(data):
            return payload.get("embeddings", [])

    os.makedirs(os.path.dirname(EMBED_CACHE_PATH), exist_ok=True)
    texts = [_build_embedding_text(item) for item in data]
    embeddings = _embed_texts(texts)
    with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"count": len(data), "embeddings": embeddings}, f)
    return embeddings


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def retrieve_top_k(
    data: List[Dict], embeddings: List[List[float]], query: str, top_k: int
) -> List[Tuple[Dict, float]]:
    query_emb = _embed_texts([query])[0]
    scored = []
    for idx, emb in enumerate(embeddings):
        score = _cosine_sim(query_emb, emb)
        scored.append((score, idx))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(data[idx], score) for score, idx in scored[:top_k]]


def render_related_inscriptions(items: List[Tuple[Dict, float]]):
    st.subheader("関連する可能性のある碑文")
    st.caption("質問に直接合致する碑文が少ないため、類似度の高い碑文を提示します。")
    for item, score in items:
        _id = item.get("id", "unknown")
        header = f"[ID: {_id}] score={score:.3f} {item.get('region', '')} ({item.get('date_min', '')}–{item.get('date_max', '')})"
        with st.expander(header):
            st.write(item.get("text", ""))


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

        st.subheader("検索語 (ギリシア語)")
        manual_terms = st.text_input(
            "カンマ区切りで入力 (例: βασιλεύς, βασιλέως, βασιλεῖ)", ""
        )
        greek_terms = [t.strip() for t in manual_terms.split(",") if t.strip()]

        normalize = st.checkbox("年代別総碑文数で正規化する", value=True, key="normalize_trend")
        search_clicked = st.button("検索を実行", key="search_btn")

        if "trend_results" not in st.session_state:
            st.session_state.trend_results = []
            st.session_state.trend_terms = []
            st.session_state.trend_rows = []

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

                st.session_state.trend_results = results
                st.session_state.trend_terms = greek_terms
                st.session_state.trend_rows = rows

        if st.session_state.trend_results:
            fig = px.line(st.session_state.trend_rows, x="year", y="value", title="Temporal Trend")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Hit List")
            for item in st.session_state.trend_results:
                _id = item.get("id", "unknown")
                header = f"[ID: {_id}] {item.get('region', '')} ({item.get('date_min', '')}–{item.get('date_max', '')})"
                translate_key = f"translate_{_id}"
                cache_key = f"jp_translation_{_id}"
                error_key = f"jp_translation_error_{_id}"
                open_key = f"open_{_id}"

                if "last_translated_id" not in st.session_state:
                    st.session_state["last_translated_id"] = None
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = ""
                if error_key not in st.session_state:
                    st.session_state[error_key] = ""
                if open_key not in st.session_state:
                    st.session_state[open_key] = False

                expanded = (
                    st.session_state[open_key]
                    or bool(st.session_state[cache_key])
                    or st.session_state["last_translated_id"] == _id
                )
                with st.expander(header, expanded=expanded):
                    if st.button("この碑文を日本語訳", key=translate_key):
                        st.session_state[open_key] = True
                        st.session_state["last_translated_id"] = _id
                        st.session_state[error_key] = ""
                        with st.spinner("日本語訳を生成中..."):
                            try:
                                st.session_state[cache_key] = translate_to_japanese(
                                    str(item.get("text", ""))
                                )
                            except Exception as e:
                                st.session_state[error_key] = str(e)

                    st.json({"metadata": item.get("metadata")})

                    if st.session_state[error_key]:
                        st.error(st.session_state[error_key])

                    if st.session_state[cache_key]:
                        st.markdown("**日本語訳:**")
                        st.write(st.session_state[cache_key])

                    st.write(item.get("text", ""))

    with tab_chat:
        st.header("碑文ベースのチャット")
        st.caption("回答は必ず [ID: 12345] の形式でエビデンスを付与します。")

        st.subheader("埋め込み検索")
        st.caption("初回のみ全碑文の埋め込み作成が必要です。時間とAPIコストがかかります。")
        top_k = st.slider("参照する碑文数 (Top-K)", min_value=5, max_value=50, value=20)
        min_score = st.slider("一致度しきい値", min_value=0.0, max_value=0.6, value=0.25, step=0.05)
        build_clicked = st.button("埋め込みを準備する", key="embed_build")

        if "embeddings_ready" not in st.session_state:
            st.session_state.embeddings_ready = False

        if build_clicked or not st.session_state.embeddings_ready:
            with st.status("Building / loading embeddings...", expanded=True) as status:
                embeddings = load_or_create_embeddings(data)
                st.session_state.embeddings = embeddings
                st.session_state.embeddings_ready = True
                status.update(label=f"Embeddings ready: {len(embeddings):,}", state="complete")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        messages_box = st.container(height=420)
        with messages_box:
            for m in st.session_state.chat_messages:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

        user_message = st.chat_input("碑文に関する質問を入力")
        if user_message:
            st.session_state.chat_messages.append({"role": "user", "content": user_message})
            with messages_box:
                with st.chat_message("user"):
                    st.write(user_message)

            with st.spinner("埋め込み検索中..."):
                if not st.session_state.embeddings_ready:
                    embeddings = load_or_create_embeddings(data)
                    st.session_state.embeddings = embeddings
                    st.session_state.embeddings_ready = True
                retrieved_scored = retrieve_top_k(
                    data, st.session_state.embeddings, user_message, top_k
                )
            retrieved = [item for item, _ in retrieved_scored]
            context = build_context(retrieved, max_items=top_k)
            if retrieved_scored and retrieved_scored[0][1] < min_score:
                render_related_inscriptions(retrieved_scored[: min(top_k, 10)])
            st.subheader("参照した碑文リスト")
            for item, score in retrieved_scored:
                _id = item.get("id", "unknown")
                header = f"[ID: {_id}] score={score:.3f} {item.get('region', '')} ({item.get('date_min', '')}–{item.get('date_max', '')})"
                with st.expander(header):
                    st.write(item.get("text", ""))
            with messages_box:
                with st.chat_message("assistant"):
                    with st.spinner("回答生成中..."):
                        answer = run_rag_chat(context, user_message)
                        st.write(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
