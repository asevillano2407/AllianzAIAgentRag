"""Streamlit interview demo for the claims copilot."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Claims Copilot", page_icon="🛡️", layout="wide")
st.title("Claims Copilot")
st.caption("CIDE, ASCIDE and CICOS evidence assistant")

with st.sidebar:
    st.subheader("Demo guardrails")
    st.write(
        "The assistant cites the supplied 2004 manual and separates agreement "
        "responsibility from legal or coverage decisions."
    )
    st.warning("A claims specialist must review every result.")

examples = {
    "Choose an example": "",
    "Rear-end at a red light": (
        "El vehiculo A esta detenido ante un semaforo rojo. El vehiculo B no "
        "frena a tiempo y alcanza por detras al vehiculo A. B afirma que A freno de golpe."
    ),
    "Lane change": (
        "El vehiculo A cambia de carril y golpea lateralmente al vehiculo B. "
        "A dice que B estaba en su angulo muerto y B dice que A no miro los espejos."
    ),
    "Alcohol": (
        "El vehiculo B golpea al vehiculo A. Despues se confirma que el conductor "
        "de B conducia bajo los efectos del alcohol. Hay lesiones graves."
    ),
}

selected = st.selectbox("Example", examples)
query = st.text_area(
    "Question or accident description",
    value=examples[selected],
    height=150,
    placeholder="Describe the facts and the available evidence...",
)

if st.button("Analyse", type="primary", disabled=len(query.strip()) < 5):
    with st.spinner("Retrieving evidence and validating the answer..."):
        try:
            response = httpx.post(
                f"{API_URL}/v1/analyze", json={"query": query}, timeout=90.0
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            st.error(f"The API could not complete the analysis: {exc}")
        else:
            analysis = payload["analysis"]
            st.subheader("Conclusion")
            st.write(analysis["answer"])

            left, right = st.columns(2)
            with left:
                st.markdown("**Responsibility under the agreements**")
                st.write(analysis["agreement_responsibility"])
                st.markdown("**Key facts**")
                for fact in analysis["key_facts"]:
                    st.write(f"- {fact}")
            with right:
                st.markdown("**Applicable framework**")
                for item in analysis["applicable_framework"]:
                    st.write(f"- {item}")
                st.markdown(f"**Confidence:** {analysis['confidence']}")

            if analysis["missing_information"]:
                st.subheader("Missing information")
                for item in analysis["missing_information"]:
                    st.write(f"- {item}")

            st.subheader("Evidence")
            for citation in analysis["citations"]:
                with st.expander(
                    f"{citation['source']} - PDF page {citation['page']}"
                ):
                    st.write(citation["quote"] or "Citation without direct quote")
                    st.code(citation["chunk_id"], language=None)

            for warning in payload["warnings"]:
                st.warning(warning)
