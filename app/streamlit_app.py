"""Live demo: Jev (TypeSafe AI) vs GPT-5.4 on Microsoft Foundry, side by side.

Run with:  poetry run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import streamlit as st

from jev_benchmark.config import REPORTS_DIR, get_settings
from jev_benchmark.systems import build_system
from jev_benchmark.systems.base import Prediction
from jev_benchmark.tasks import TASKS, Example, Task, TaskKind

st.set_page_config(page_title="Jev vs GPT-5.4", page_icon="⚖️", layout="wide")

SYSTEMS = ["jev", "gpt-5.4-none", "gpt-5.4-medium"]
LABELS = {
    "jev": "Jev",
    "gpt-5.4-none": "GPT-5.4 · reasoning none",
    "gpt-5.4-medium": "GPT-5.4 · reasoning medium",
}
COLORS = {"jev": "#2a78d6", "gpt-5.4-none": "#eb6834", "gpt-5.4-medium": "#1baf7a"}  # same as the report

settings = get_settings()
JEV_READY = bool(settings.typesafe_api_key.get_secret_value())


async def classify_all(names: list[str], task: Task, example: Example) -> dict[str, Prediction | str]:
    async def one(name: str) -> Prediction | str:
        try:
            system = build_system(name, settings)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            return f"{type(exc).__name__}: {exc}"
        try:
            return await system.classify(task, example)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        finally:
            await system.aclose()

    results = await asyncio.gather(*(one(n) for n in names))
    return dict(zip(names, results, strict=True))


def signal_of(task: Task, pred: Prediction) -> float | None:
    """Same escalation signal used in the benchmark cascade."""
    return pred.p_pred if task.kind is TaskKind.CHOICE else pred.confidence


# --- sidebar --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Configurazione")
    task_name = st.selectbox("Scenario", list(TASKS), format_func=lambda n: TASKS[n].title)
    task = TASKS[task_name]
    default = ["jev", "gpt-5.4-none"] if JEV_READY else ["gpt-5.4-none"]
    chosen = st.multiselect("Sistemi da confrontare", SYSTEMS, default=default, format_func=LABELS.get)
    tau = st.slider(
        "Soglia della cascata τ",
        0.0,
        1.0,
        0.80,
        0.01,
        help="Se la probabilità (S1, S2) o la confidenza (S3) di Jev è sotto τ, la richiesta passa all'LLM.",
    )
    fallback = st.selectbox("LLM di fallback", SYSTEMS[1:], format_func=LABELS.get)
    if not JEV_READY:
        st.warning("`TYPESAFE_API_KEY` non impostata in `.env`: Jev non è disponibile.")
    st.caption(f"Jev: `{settings.jev_model}` · Foundry: `{settings.azure_openai_deployment}`")

st.title("Jev vs GPT-5.4 su Microsoft Foundry")
st.caption(f"{task.title} · {len(task.labels)} opzioni · dataset: {task.dataset}")

tab_live, tab_results = st.tabs(["Prova dal vivo", "Risultati del benchmark"])

# --- live tab -------------------------------------------------------------------------------------
with tab_live:
    examples = task.load()
    mode = st.radio("Input", ["Esempio dal benchmark", "Testo libero"], horizontal=True)
    if mode == "Esempio dal benchmark":
        idx = st.number_input("Esempio n°", 0, len(examples) - 1, 0, 1)
        example = examples[int(idx)]
        gold: str | None = example.label
        for field, value in example.state.items():
            st.text_area(field, value, height=220 if len(value) > 400 else 90, disabled=True)
        st.markdown(f"Risposta attesa (gold): **`{gold}`**")
    else:
        gold = None
        state = {
            field: st.text_area(field, "", height=160, placeholder=f"Inserisci {field}…")
            for field in examples[0].state
        }
        example = Example(id="custom", label=task.label_names[0], state=state)

    with st.expander("Opzioni e descrizioni (identiche per entrambi i sistemi)"):
        st.markdown(f"**Domanda:** {task.instructions}")
        st.dataframe(
            pd.DataFrame({"opzione": list(task.labels), "descrizione": list(task.labels.values())}),
            hide_index=True,
            width="stretch",
        )

    if st.button("Classifica", type="primary", disabled=not chosen or not all(example.state.values())):
        names = list(dict.fromkeys(chosen + (["jev", fallback] if JEV_READY else [])))
        with st.spinner("Chiamata ai sistemi in parallelo…"):
            results = asyncio.run(classify_all(names, task, example))

        cols = st.columns(len(chosen))
        for col, name in zip(cols, chosen, strict=True):
            res = results[name]
            with col:
                st.subheader(LABELS[name])
                if isinstance(res, str):
                    st.error(res)
                    continue
                verdict = "" if gold is None else (" ✅" if res.pred == gold else " ❌")
                st.metric("Decisione", f"{res.pred}{verdict}")
                c1, c2, c3 = st.columns(3)
                c1.metric("p(decisione)", "–" if res.p_pred is None else f"{res.p_pred:.2f}")
                c2.metric("Latenza", f"{res.latency_ms:,.0f} ms")
                c3.metric("Costo", f"${res.cost_usd * 1000:.3f}/1k")
                if res.probabilities:
                    top = sorted(res.probabilities.items(), key=lambda kv: -kv[1])[:10]
                    chart = pd.DataFrame(top, columns=["opzione", "probabilità"]).set_index("opzione")
                    st.bar_chart(chart, horizontal=True, color=COLORS[name], height=260)
                st.caption(
                    f"modello `{res.model}` · token input {res.input_tokens:,}"
                    + (f" (cache {res.cached_tokens:,})" if res.cached_tokens else "")
                    + (f" · reasoning {res.reasoning_tokens:,}" if res.reasoning_tokens else "")
                )

        jev, llm = results.get("jev"), results.get(fallback)
        if isinstance(jev, Prediction):
            st.divider()
            st.subheader("Cascata Jev → LLM")
            signal = signal_of(task, jev)
            if signal is not None and signal >= tau:
                st.success(
                    f"Jev è sicuro ({signal:.2f} ≥ τ = {tau:.2f}): si usa **{jev.pred}** "
                    f"in {jev.latency_ms:,.0f} ms, senza chiamare l'LLM."
                )
            elif isinstance(llm, Prediction):
                total_ms = jev.latency_ms + llm.latency_ms
                st.info(
                    f"Jev è incerto ({signal:.2f} < τ = {tau:.2f}): la richiesta passa a {LABELS[fallback]}, "
                    f"che risponde **{llm.pred}**. Latenza totale ≈ {total_ms:,.0f} ms."
                )

# --- results tab ----------------------------------------------------------------------------------
with tab_results:
    summary_path = REPORTS_DIR / "main" / "summary.json"
    if not summary_path.exists():
        st.info("Nessun report: esegui `jevbench run` e poi `jevbench report`.")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = []
        for name, entry in summary["tasks"].items():
            for system, s in entry["systems"].items():
                rows.append(
                    {
                        "scenario": TASKS[name].title,
                        "sistema": LABELS.get(system, system),
                        "accuratezza": s["quality"]["accuracy"],
                        "latenza p50 (ms)": s["latency"]["p50_ms"],
                        "latenza p95 (ms)": s["latency"]["p95_ms"],
                        "USD / 1.000 richieste": s["cost"]["usd_per_1k"],
                    }
                )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "accuratezza": st.column_config.NumberColumn(format="percent"),
                "latenza p50 (ms)": st.column_config.NumberColumn(format="%.0f"),
                "latenza p95 (ms)": st.column_config.NumberColumn(format="%.0f"),
                "USD / 1.000 richieste": st.column_config.NumberColumn(format="$%.4f"),
            },
        )
        st.caption(
            f"Run `main`, generato il {summary['generated_at']} · dettagli in `reports/main/report.md`"
        )
        for fig in ("quality", "tradeoff", "latency", "calibration", "selective", "confusion"):
            path = REPORTS_DIR / "main" / "figures" / f"{fig}-light.png"
            if path.exists():
                st.image(str(path), width="stretch")
