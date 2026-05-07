import os
import requests
import streamlit as st


st.set_page_config(page_title="Enterprise AI Assistant Demo", layout="wide")

st.title("Enterprise AI Assistant")
st.caption("RAG + simple agent orchestration + tool calling (calculator).")

backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

question = st.text_area("Ask a question", placeholder="e.g. What does our policy say about refunds?\nOr: calc: (12*3)+5")
top = st.button("Query", type="primary")

col1, col2 = st.columns([2, 1], gap="large")

if top and question.strip():
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{backend_url}/query", json={"question": question}, timeout=120)
            if resp.status_code != 200:
                st.error(f"Backend error ({resp.status_code}): {resp.text}")
            else:
                data = resp.json()
                with col1:
                    st.subheader("Answer")
                    steps = data.get("steps", []) or []
                    if any("retrieve: skipped (no index found)" in s for s in steps):
                        st.warning(
                            "No RAG index found yet. Running in normal chat mode (no database/context retrieval). "
                            "Run the ingestion step to enable RAG."
                        )
                    st.write(data.get("answer", ""))

                    st.subheader("Retrieved context")
                    ctx = data.get("context", [])
                    if not ctx:
                        st.write("(none)")
                    else:
                        for i, c in enumerate(ctx, start=1):
                            with st.expander(f"Chunk {i}"):
                                st.write(c)

                with col2:
                    st.subheader("Agent steps")
                    for s in steps:
                        st.code(s)
        except Exception as e:  # noqa: BLE001
            st.exception(e)
else:
    st.info(f"Backend: {backend_url}")

