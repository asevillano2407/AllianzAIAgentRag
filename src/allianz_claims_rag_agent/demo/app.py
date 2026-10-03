"""Streamlit interface for replaying results or invoking the local agent."""

from pathlib import Path
from typing import Any

from allianz_claims_rag_agent.config import Settings
from allianz_claims_rag_agent.demo.service import (
    load_saved_demo_runs,
    serialize_agent_result,
)
from allianz_claims_rag_agent.errors import ApplicationError
from allianz_claims_rag_agent.orchestration import run_local_agent

LOGO_PATH = Path(__file__).with_name("assets") / "allianz_logo.svg"


def main() -> None:
    """Render the local demonstration application."""
    import streamlit as st

    st.set_page_config(
        page_title="Allianz Claims RAG",
        page_icon="🛡️",
        layout="wide",
    )
    logo_column, title_column = st.columns([1, 4], vertical_alignment="center")
    with logo_column:
        st.image(str(LOGO_PATH), width=190)
    with title_column:
        st.title("Claims RAG Agent")
        st.caption(
            "Análisis local del manual CIDE/ASCIDE/CICOS · "
            "Qdrant + Ollama + LangGraph"
        )

    mode = st.sidebar.radio(
        "Modo de demostración",
        ["Resultado evaluado", "Ejecutar agente real"],
        help="El resultado evaluado es instantáneo; el agente real puede tardar varios minutos.",
    )
    st.sidebar.markdown("### Configuración seleccionada")
    st.sidebar.code(
        "Embedding: qwen3-embedding:0.6b\n"
        "Reranker: BAAI/bge-reranker-v2-m3\n"
        "Candidatos: 12 · Contexto: 3\n"
        "LLM: qwen3:4b",
        language=None,
    )
    st.sidebar.caption("Todo se ejecuta en local y sin servicios de pago.")
    st.sidebar.caption("Prototipo técnico no oficial · Logo: Allianz SE")

    if mode == "Resultado evaluado":
        _render_saved_mode(st)
    else:
        _render_live_mode(st)


def _render_saved_mode(st: Any) -> None:
    runs = load_saved_demo_runs()
    runs_by_title = {run["title"]: run for run in runs}
    selected_title = st.selectbox("Caso evaluado", list(runs_by_title))
    selected = runs_by_title[selected_title]
    st.info(
        "Resultado real guardado del benchmark final. No se ejecuta el LLM en este modo."
    )
    st.text_area("Relato", selected["query"], height=145, disabled=True)
    _render_result(st, selected)


def _render_live_mode(st: Any) -> None:
    st.warning(
        "La inferencia con qwen3:4b tarda aproximadamente 6–9 minutos en la CPU "
        "del equipo evaluado. Mantén Ollama y el índice Qdrant disponibles."
    )
    query = st.text_area(
        "Describe el accidente o formula una pregunta sobre el manual",
        value=(
            "Al cambiar de carril en la autopista, el vehículo A roza lateralmente "
            "al vehículo B."
        ),
        height=145,
        max_chars=4_000,
    )
    use_reranker = st.checkbox("Aplicar reranking híbrido", value=True)
    if st.button("Ejecutar análisis", type="primary", disabled=not query.strip()):
        try:
            with st.spinner("Ejecutando retrieval, reranking y generación local..."):
                result = run_local_agent(
                    query,
                    settings=Settings(ollama_timeout_seconds=600),
                    rerank=use_reranker,
                    top_k=3,
                    candidate_k=12,
                )
            st.session_state["allianz_live_result"] = serialize_agent_result(
                query,
                result,
            )
        except (ApplicationError, ValueError, OSError) as exc:
            st.error(f"No se pudo completar el análisis local: {exc}")

    live_result = st.session_state.get("allianz_live_result")
    if live_result is not None:
        _render_result(st, live_result)


def _render_result(st: Any, run: dict[str, Any]) -> None:
    response = run["response"]
    st.subheader("Decisión")
    applicability, responsibility, confidence = st.columns(3)
    applicability.metric(
        "Aplicabilidad CIDE/ASCIDE",
        _label(response.get("convention_applicability")),
    )
    responsibility.metric(
        "Responsabilidad según convenio",
        _label(response.get("convention_responsibility")),
    )
    confidence.metric("Confianza", _label(response.get("confidence")))
    st.success(response["conclusion"])

    facts_column, missing_column = st.columns(2)
    with facts_column:
        st.markdown("#### Hechos considerados")
        _render_items(st, response.get("facts", []), "No se extrajeron hechos.")
    with missing_column:
        st.markdown("#### Información ausente")
        _render_items(
            st,
            response.get("missing_information", []),
            "No se indicó información ausente.",
        )

    st.markdown("#### Evidencia citada")
    citation_groups = [
        ("Aplicabilidad", response.get("applicability_citations", [])),
        ("Responsabilidad", response.get("responsibility_citations", [])),
        ("General", response.get("citations", [])),
    ]
    if not any(citations for _, citations in citation_groups):
        st.caption("La respuesta no contiene citas.")
    for group_name, citations in citation_groups:
        for index, citation in enumerate(citations, 1):
            with st.expander(
                f"{group_name} · página {citation['page']} · cita {index}"
            ):
                st.write(citation["quote"])
                st.caption(f"Chunk: {citation['chunk_id']}")

    evaluation = run.get("business_evaluation")
    if evaluation:
        st.markdown("#### Evaluación automática")
        columns = st.columns(3)
        columns[0].metric(
            "Aplicabilidad",
            _correct_label(evaluation.get("applicability_correct")),
        )
        columns[1].metric(
            "Responsabilidad",
            _correct_label(evaluation.get("responsibility_correct")),
        )
        columns[2].metric(
            "Caso completo",
            _correct_label(evaluation.get("business_correct")),
        )

    with st.expander("Detalle técnico"):
        agent = run.get("agent") or {}
        generation = run.get("generation") or {}
        retrieval = run.get("retrieval") or {}
        st.write(
            {
                "origen": run.get("source"),
                "estado": agent.get("status"),
                "tipo_consulta": agent.get("query_type"),
                "recorrido_langgraph": agent.get("execution_path"),
                "fallback": agent.get("fallback_used"),
                "reintentos": agent.get("retry_count"),
                "paginas_recuperadas": retrieval.get("pages"),
                "consultas_retrieval": retrieval.get("queries"),
                "ajustes_fail_soft": generation.get("adjustments", []),
                "modelo": generation.get("model_name"),
                "prompt_tokens": generation.get("prompt_tokens"),
                "completion_tokens": generation.get("completion_tokens"),
                "duracion_segundos": _duration_seconds(
                    generation.get("total_duration_ms")
                ),
            }
        )


def _render_items(st: Any, items: list[str], empty_message: str) -> None:
    if not items:
        st.caption(empty_message)
        return
    for item in items:
        st.markdown(f"- {item}")


def _label(value: str | None) -> str:
    labels = {
        "applicable": "Aplicable",
        "not_applicable": "No aplicable",
        "undetermined": "Indeterminada",
        "vehicle_a": "Vehículo A",
        "vehicle_b": "Vehículo B",
        "shared": "Compartida",
        "low": "Baja",
        "medium": "Media",
        "high": "Alta",
    }
    return labels.get(value, value or "No aplica")


def _correct_label(value: bool | None) -> str:
    return "Correcta" if value else "Incorrecta"


def _duration_seconds(value: int | float | None) -> float | None:
    return round(value / 1_000, 1) if value is not None else None


if __name__ == "__main__":
    main()
