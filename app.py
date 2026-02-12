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
try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency at runtime
    chromadb = None


DATA_PATH = "egypt_processed_tagged.json"
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHROMA_DIR = ".chroma"
CHROMA_COLLECTION = "inscriptions"
TRANSLATE_MODEL = "gpt-4o"


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


def search_inscriptions_text(
    data: List[Dict], terms: List[str], year_range: Tuple[int, int] | None = None
) -> List[Tuple[Dict, float]]:
    rx = _build_regex(terms)
    results: List[Tuple[Dict, float]] = []
    for item in data:
        dmin = _safe_int(item.get("date_min"))
        dmax = _safe_int(item.get("date_max"))
        if year_range is not None and dmin is not None and dmax is not None:
            start, end = year_range
            if not (dmin <= end and dmax >= start):
                continue

        text = str(item.get("text", ""))
        lemmas = item.get("lemmas", [])
        keywords = item.get("keywords", [])
        hay = " ".join(
            [
                text,
                " ".join(str(x) for x in lemmas) if isinstance(lemmas, list) else str(lemmas),
                " ".join(str(x) for x in keywords) if isinstance(keywords, list) else str(keywords),
            ]
        )
        matches = len(rx.findall(hay))
        if matches > 0:
            results.append((item, float(matches)))
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


@st.cache_data(show_spinner=False)
def get_date_bounds(data: List[Dict]) -> Tuple[int, int]:
    mins = []
    maxs = []
    for item in data:
        dmin = _safe_int(item.get("date_min"))
        dmax = _safe_int(item.get("date_max"))
        if dmin is not None:
            mins.append(dmin)
        if dmax is not None:
            maxs.append(dmax)
    if not mins or not maxs:
        return -3000, 1000
    return min(mins), max(maxs)


def count_in_range(data: List[Dict], year_range: Tuple[int, int]) -> int:
    start, end = year_range
    count = 0
    for item in data:
        dmin = _safe_int(item.get("date_min"))
        dmax = _safe_int(item.get("date_max"))
        if dmin is None or dmax is None:
            continue
        if dmin <= end and dmax >= start:
            count += 1
    return count


def run_rag_chat(context: str, user_message: str, allow_background: bool, answer_len: int) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    client = OpenAI(api_key=api_key)

    length_hint = {
        1: "Respond in 1-2 short sentences.",
        2: "Respond in a short paragraph (3-4 sentences).",
        3: "Respond in a medium-length answer (5-7 sentences).",
        4: "Respond in a detailed answer (8-12 sentences).",
        5: "Respond in a very detailed answer with structured paragraphs.",
    }.get(answer_len, "Respond in a medium-length answer (5-7 sentences).")

    if allow_background:
        system_prompt = (
            "You are a careful scholar. Use the provided inscriptions context first. "
            "You may add general historical background knowledge if it is clearly marked as such. "
            "Rules: "
            "1) For statements grounded in inscriptions, cite evidence after each statement in the format [ID: 12345]. "
            "2) For statements from general knowledge, prefix the sentence with '背景知識:' and do NOT cite inscription IDs. "
            "3) Do not fabricate inscription content. If the context is insufficient, say so explicitly."
            f" {length_hint}"
        )
    else:
        system_prompt = (
            "You are a careful scholar. Answer ONLY from the provided inscriptions context. "
            "After each factual statement, cite evidence in the format [ID: 12345]. "
            "If the context is insufficient, say so explicitly and do not speculate."
            f" {length_hint}"
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
        "You are a skilled translator of Ancient Greek inscriptions. "
        "Provide a natural, readable Japanese translation while staying faithful to the text. "
        "Rules: "
        "1) Preserve proper names and technical terms; use standard scholarly transliterations. "
        "2) If a segment is unclear, mark with （不明） or （不確実）. "
        "3) Keep line order if possible, but prioritize readability. "
        "Return ONLY the Japanese translation, no extra commentary."
    )
    response = client.chat.completions.create(
        model=TRANSLATE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def expand_query_for_search(user_query: str) -> List[str]:
    # Build multiple query variants (JP -> EN/Greek) to improve recall.
    api_key = _get_api_key()
    if not api_key or OpenAI is None:
        return [user_query]

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a search assistant for ancient inscriptions. "
        "Return ONLY JSON with keys: english_query, greek_keywords. "
        "english_query: a short English paraphrase of the user query. "
        "greek_keywords: a short list of relevant Ancient Greek lemmas or terms."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        english_query = str(data.get("english_query", "")).strip()
        greek_keywords = data.get("greek_keywords", [])
        if isinstance(greek_keywords, list):
            greek_text = " ".join(str(x) for x in greek_keywords)
        else:
            greek_text = str(greek_keywords)
        queries = [user_query]
        if english_query:
            queries.append(english_query)
        if greek_text.strip():
            queries.append(greek_text.strip())
        return list(dict.fromkeys([q for q in queries if q]))
    except Exception:
        return [user_query]


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if OpenAI is None:
        raise RuntimeError("openai package is not available")

    client = OpenAI(api_key=api_key)
    embeddings: List[List[float]] = []
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch = [_truncate_text(t) for t in texts[i : i + batch_size]]
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


def _get_chroma_collection():
    if chromadb is None:
        raise RuntimeError("chromadb package is not available")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(CHROMA_COLLECTION)


def _needs_rebuild(collection) -> bool:
    try:
        peek = collection.get(limit=1, include=["metadatas"])
        metas = peek.get("metadatas", [[]])[0]
        if not metas:
            return True
        meta = metas[0]
        return "date_min" not in meta or "date_max" not in meta
    except Exception:
        return True


def build_chroma_index(data: List[Dict]):
    collection = _get_chroma_collection()
    existing = collection.count()
    if existing == len(data) and not _needs_rebuild(collection):
        return collection

    # Rebuild if counts don't match
    try:
        collection.delete(where={})
    except Exception:
        pass

    texts = [_build_embedding_text(item) for item in data]
    embeddings = _embed_texts(texts)
    ids = [str(item.get("id", idx)) for idx, item in enumerate(data)]
    metadatas = []
    for idx, item in enumerate(data):
        metadatas.append(
            {
                "idx": idx,
                "date_min": _safe_int(item.get("date_min")),
                "date_max": _safe_int(item.get("date_max")),
            }
        )

    batch_size = 1000
    for i in range(0, len(data), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            documents=texts[i : i + batch_size],
        )
    return collection


def chroma_query(
    data: List[Dict],
    collection,
    query: str,
    top_k: int,
    date_range: Tuple[int, int] | None = None,
) -> List[Tuple[Dict, float]]:
    query_emb = _embed_texts([query])[0]
    where = None
    if date_range is not None:
        start, end = date_range
        where = {"$and": [{"date_min": {"$lte": end}}, {"date_max": {"$gte": start}}]}
    res = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["metadatas", "distances"],
        where=where,
    )
    items: List[Tuple[Dict, float]] = []
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    for meta, dist in zip(metadatas, distances):
        idx = meta.get("idx")
        if idx is None:
            continue
        score = 1.0 / (1.0 + float(dist))
        items.append((data[int(idx)], score))
    return items


def multi_query_retrieve(
    data: List[Dict],
    collection,
    queries: List[str],
    top_k: int,
    date_range: Tuple[int, int] | None = None,
) -> List[Tuple[Dict, float]]:
    merged: Dict[str, Tuple[Dict, float]] = {}
    for q in queries:
        results = chroma_query(data, collection, q, top_k, date_range)
        for item, score in results:
            _id = str(item.get("id", "unknown"))
            if _id not in merged or score > merged[_id][1]:
                merged[_id] = (item, score)
    ordered = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


def hybrid_retrieve(
    data: List[Dict],
    collection,
    queries: List[str],
    top_k: int,
    date_range: Tuple[int, int] | None = None,
    alpha: float = 0.7,
) -> List[Tuple[Dict, float]]:
    vec = multi_query_retrieve(data, collection, queries, top_k, date_range)
    text = search_inscriptions_text(data, queries, year_range=date_range)

    vec_map: Dict[str, Tuple[Dict, float]] = {str(item.get("id", "unknown")): (item, score) for item, score in vec}
    if text:
        max_text = max(score for _, score in text) or 1.0
    else:
        max_text = 1.0
    text_map: Dict[str, Tuple[Dict, float]] = {}
    for item, score in text:
        _id = str(item.get("id", "unknown"))
        text_map[_id] = (item, score / max_text)

    merged: Dict[str, Tuple[Dict, float]] = {}
    ids = set(vec_map.keys()) | set(text_map.keys())
    for _id in ids:
        item = vec_map.get(_id, text_map.get(_id))[0]
        v = vec_map.get(_id, (item, 0.0))[1]
        t = text_map.get(_id, (item, 0.0))[1]
        score = alpha * v + (1.0 - alpha) * t
        merged[_id] = (item, score)

    ordered = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


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
                _id = str(item.get("id", "unknown"))
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
                if st.session_state[cache_key]:
                    st.session_state[open_key] = True

                open_now = st.checkbox(f"▼ {header}", value=st.session_state[open_key], key=f"toggle_{_id}")
                st.session_state[open_key] = open_now
                if open_now:
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

        st.subheader("埋め込み検索 (ChromaDB)")
        st.caption("初回のみ全碑文の埋め込み作成が必要です。以後は高速検索できます。")
        # 年代フィルタを一旦無効化（常に全件対象）
        top_k = st.slider("参照する碑文数 (Top-K)", min_value=5, max_value=50, value=20)
        min_score = st.slider("一致度しきい値", min_value=0.0, max_value=0.6, value=0.25, step=0.05)
        build_clicked = st.button("ChromaDBを準備する", key="embed_build")
        expand_query = st.checkbox("日本語クエリを拡張して検索精度を上げる", value=True)
        use_hybrid = st.checkbox("ベクトル+文字列のハイブリッド検索", value=True)
        alpha = st.slider("ベクトル寄りの重み", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        allow_background = st.checkbox("背景知識モード（一般的な歴史知識も許可）", value=False)
        answer_len = st.slider("回答の長さ", min_value=1, max_value=5, value=3)

        if "chroma_ready" not in st.session_state:
            st.session_state.chroma_ready = False

        if build_clicked or not st.session_state.chroma_ready:
            with st.status("Building / loading ChromaDB index...", expanded=True) as status:
                collection = build_chroma_index(data)
                st.session_state.chroma_collection = collection
                st.session_state.chroma_ready = True
                status.update(label=f"ChromaDB ready: {collection.count():,}", state="complete")

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
                if not st.session_state.chroma_ready:
                    collection = build_chroma_index(data)
                    st.session_state.chroma_collection = collection
                    st.session_state.chroma_ready = True
                queries = [user_message]
                if expand_query:
                    queries = expand_query_for_search(user_message)
                if use_hybrid:
                    retrieved_scored = hybrid_retrieve(
                        data, st.session_state.chroma_collection, queries, top_k, None, alpha
                    )
                else:
                    retrieved_scored = multi_query_retrieve(
                        data, st.session_state.chroma_collection, queries, top_k, None
                    )
            retrieved = [item for item, _ in retrieved_scored]
            context = build_context(retrieved, max_items=top_k)
            if retrieved_scored and retrieved_scored[0][1] < min_score:
                render_related_inscriptions(retrieved_scored[: min(top_k, 10)])
            if not retrieved_scored:
                st.warning("指定した年代範囲に該当する碑文がありません。年代範囲を広げてください。")
            st.subheader("参照した碑文リスト")
            for item, score in retrieved_scored:
                _id = item.get("id", "unknown")
                header = f"[ID: {_id}] score={score:.3f} {item.get('region', '')} ({item.get('date_min', '')}–{item.get('date_max', '')})"
                with st.expander(header):
                    st.write(item.get("text", ""))
            with messages_box:
                with st.chat_message("assistant"):
                    with st.spinner("回答生成中..."):
                        answer = run_rag_chat(context, user_message, allow_background, answer_len)
                        st.write(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
