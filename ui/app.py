"""Streamlit A/B UI: base vs fine-tuned, side by side.

Point the two endpoint URLs at a base-model server and a fine-tuned server
(run scripts/serve.py twice — once with --adapter, once without). Enter a
question + db_id and compare the generated SQL and latency. Single-endpoint mode
works too (leave the second URL blank).

    streamlit run ui/app.py
"""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="text2sql-qlora — A/B", layout="wide")
st.title("Text-to-SQL: base vs fine-tuned")

with st.sidebar:
    st.header("Endpoints")
    url_a = st.text_input("A — base", "http://localhost:8001")
    url_b = st.text_input("B — fine-tuned", "http://localhost:8000")
    st.caption("Run scripts/serve.py twice (with/without --adapter) on two ports.")

db_id = st.text_input("Database (db_id)", "concert_singer")
question = st.text_area("Question", "How many singers do we have?", height=80)
go = st.button("Generate", type="primary")


def call(url: str, question: str, db_id: str) -> dict:
    r = requests.post(
        f"{url.rstrip('/')}/generate",
        json={"question": question, "db_id": db_id, "include_raw": True},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def render(col, label: str, url: str):
    with col:
        st.subheader(label)
        if not url:
            st.info("no endpoint")
            return
        try:
            out = call(url, question, db_id)
            st.code(out["sql"] or "(empty)", language="sql")
            st.caption(
                f"{out['latency_ms']:.0f} ms · "
                f"{out['prompt_tokens']}→{out['completion_tokens']} tok · {out['model']}"
            )
        except Exception as e:
            st.error(str(e))


if go:
    a, b = st.columns(2)
    render(a, "A — base", url_a)
    render(b, "B — fine-tuned", url_b)