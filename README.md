# Jev vs LLM generativo su Microsoft Foundry

**Benchmark di classificazione in tre scenari.** Il PoC valuta se **Jev** di
[TypeSafe AI](https://docs.typesafe.ai), un modello di decisione che restituisce la classe scelta con
probabilità e confidenza, può sostituire o affiancare un **LLM generativo** (GPT-5.4 servito da
**Microsoft Foundry** con Structured Outputs) nelle decisioni ricorrenti tra opzioni predefinite. Le
due soluzioni sono confrontate su **qualità, latenza e costo**, sugli stessi dati e con la stessa
definizione del compito.

| Scenario | Dataset | Decisione |
|---|---|---|
| **S1** · Classificazione delle richieste in ingresso | Banking77 | 1 di 77 intent |
| **S2** · Decisione del passo successivo in un workflow | ABCD | 1 di 30 azioni, data la procedura (SOP) e la conversazione |
| **S3** · Valutazione della qualità di un output generato | HelpSteer2 | voto di helpfulness 0-4 (LLM-as-a-judge) |

## Risultati

Report completo, generato automaticamente: [reports/main/report.md](reports/main/report.md).
Interpretazione e raccomandazioni: [docs/04-risultati-e-raccomandazioni.md](docs/04-risultati-e-raccomandazioni.md).

> **Stato:** i baseline GPT-5.4 (`reasoning none` e `medium`) sono stati eseguiti su tutti i 908
> esempi. Le predizioni di **Jev** richiedono una `TYPESAFE_API_KEY` in `.env`: con la key basta
> lanciare `poetry run jevbench run --systems jev` e `poetry run jevbench report`.

## Setup

Prerequisiti: Python 3.11-3.13 e [Poetry](https://python-poetry.org) 2.x.

```bash
poetry install --with demo,dev   # crea il virtualenv in ./.venv (vedi poetry.toml)
cp .env.example .env             # poi inserisci i segreti
```

| Variabile `.env` | Descrizione |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | endpoint della risorsa Foundry, es. `https://<risorsa>.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_API_KEY` | chiave della risorsa |
| `AZURE_OPENAI_API_VERSION` | `2025-04-01-preview` (Responses API) |
| `AZURE_OPENAI_DEPLOYMENT` | nome del deployment, es. `gpt-5.4` |
| `TYPESAFE_API_KEY` | chiave Jev da [console.typesafe.ai/keys](https://console.typesafe.ai/keys) |
| `JEV_MODEL` | versione pinnata di Jev, es. `jev-1.14.0` |

`.env` e `.venv/` sono esclusi da git.

## Uso

```bash
poetry run jevbench tasks                     # elenca gli scenari
poetry run jevbench check                     # una richiesta di prova a ogni sistema
poetry run jevbench run                       # tutti i sistemi su tutti gli scenari (riprendibile)
poetry run jevbench run --systems jev --tasks intent_routing --limit 20
poetry run jevbench report                    # metriche, grafici e reports/main/report.md
poetry run jevbench classify "My card still hasn't arrived" --systems jev,gpt-5.4-none
poetry run streamlit run app/streamlit_app.py # demo interattiva
```

Nomi dei sistemi: `jev` oppure `<deployment>-<reasoning effort>` (es. `gpt-5.4-none`,
`gpt-5.4-medium`, `gpt-5.4-low`).

### Demo

La demo Streamlit permette di:

- scegliere uno scenario e un esempio del benchmark (con la risposta attesa) o scrivere un input libero;
- vedere affiancate le decisioni di Jev e di GPT-5.4, con la distribuzione di probabilità, la latenza,
  il costo per 1.000 richieste e i token;
- simulare la **cascata** Jev → LLM con una soglia τ regolabile;
- consultare i grafici e le tabelle del benchmark.

## Struttura

```
├── src/jev_benchmark/
│   ├── tasks/          # i 3 scenari: opzioni, descrizioni, preparazione dei dataset
│   ├── systems/        # jev.py (typesafe-sdk) e foundry.py (Azure OpenAI Responses API)
│   ├── runner.py       # esecuzione asincrona, retry, warm-up, output JSONL riprendibile
│   ├── metrics.py      # qualità + IC, McNemar, calibrazione, latenza, costo, cascata
│   ├── report.py       # report Markdown e grafici (tema chiaro e scuro)
│   └── cli.py          # comando `jevbench`
├── app/streamlit_app.py
├── data/benchmark/     # i 908 esempi campionati (versionati) + README con licenze
├── results/<run_id>/   # predizioni per esempio + manifest (versioni, prezzi, commit)
├── reports/<run_id>/   # report.md, summary.json, figures/
├── docs/               # documentazione (vedi sotto)
└── tests/              # test offline, inclusa l'integrazione Jev con trasporto HTTP simulato
```

## Documentazione

1. [Jev di TypeSafe AI: cos'è e come si usa](docs/01-jev-typesafe.md): API, SDK, primitive, confidenza, pattern e limiti
2. [Integrazione con Microsoft Foundry](docs/02-integrazione-foundry.md): baseline GPT-5.4 e architetture Jev + Foundry
3. [Metodologia del benchmark](docs/03-metodologia.md): dataset, parità delle condizioni, metriche, limiti
4. [Risultati e raccomandazioni](docs/04-risultati-e-raccomandazioni.md): quando usare Jev, l'LLM o entrambi

## Test

```bash
poetry run pytest          # offline: nessuna chiamata di rete
poetry run ruff check src tests
```

## Licenze dei dati

Banking77 (CC BY 4.0), ABCD (MIT) e HelpSteer2 (CC BY 4.0): dettagli e citazioni in
[data/README.md](data/README.md).
