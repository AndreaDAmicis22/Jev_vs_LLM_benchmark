# Jev vs LLM generativo su Microsoft Foundry

**Benchmark di classificazione in quattro scenari.** Il PoC valuta se **Jev** di
[TypeSafe AI](https://docs.typesafe.ai), un modello di decisione che restituisce la classe scelta con
probabilità e confidenza, può sostituire o affiancare un **LLM generativo** servito da **Microsoft
Foundry** con Structured Outputs (GPT-5.4 e GPT-5.4 mini) nelle decisioni ricorrenti tra opzioni
predefinite. Le soluzioni sono confrontate su **qualità, latenza e costo**, sugli stessi dati e con la
stessa definizione del compito.

| Scenario | Dataset | Decisione |
|---|---|---|
| **S1** · Classificazione delle richieste in ingresso | Banking77 | 1 di 77 intent |
| **S2** · Decisione del passo successivo in un workflow | ABCD | 1 di 30 azioni, data la procedura (SOP) e la conversazione |
| **S3** · Valutazione della qualità di un output generato | HelpSteer2 | voto di helpfulness 0-4 (LLM-as-a-judge) |
| **S4** · Classificazione delle richieste **in italiano** | MASSIVE it-IT | 1 di 60 intent, con istruzioni e opzioni in italiano |

500 esempi per scenario. In più, un **test del bias di posizione**: le opzioni vengono riordinate per
verificare se Jev tende a scegliere quella in cima alla lista.

| Sistema | Ruolo |
|---|---|
| `jev` | Jev (TypeSafe AI), versione pinnata `jev-1.14.0` |
| `jev-perm` | Jev con media su 3 ordini delle opzioni nella stessa richiesta (mitigazione del bias) |
| `gpt-5.4-mini-none` | **alternativa economica** su Foundry: il termine di paragone sul costo |
| `gpt-5.4-none` | riferimento di qualità e latenza |
| `gpt-5.4-medium` | tetto di qualità (reasoning medio) |
| `<sistema>@reversed`, `<sistema>@shuffled` | stesse richieste con le opzioni in ordine inverso o casuale (bias di posizione) |

## Risultati

Report completo, generato automaticamente: [reports/main/report.md](reports/main/report.md).
Interpretazione e raccomandazioni: [docs/04-risultati-e-raccomandazioni.md](docs/04-risultati-e-raccomandazioni.md).

> **Stato (13/11/2026).** I baseline GPT-5.4 (`none` e `medium`) sono stati eseguiti sul campione v1
> (908 esempi), che la v2 contiene integralmente. Il codice della v2 è pronto ma **non ancora
> eseguito**: le chiamate partiranno quando sarà disponibile la `TYPESAFE_API_KEY`.
>
> Stima offline del costo per completare la v2 (`poetry run jevbench estimate`):
>
> | Provider | Richieste | Costo stimato |
> |---|---|---|
> | TypeSafe (Jev, `jev-perm`, varianti di ordine) | 6.500 | **≈ $0,51** |
> | Microsoft Foundry (esempi nuovi di GPT-5.4, GPT-5.4 mini, varianti di ordine di GPT-5.4 none) | 7.194 | **≈ $16,3** |
>
> Circa $6,7 della spesa Foundry vengono dalla variante con ordine casuale, che non beneficia del
> prompt caching.

## Setup

Prerequisiti: Python 3.11-3.13 e [Poetry](https://python-poetry.org) 2.x.

```bash
poetry install --with demo,dev   # crea il virtualenv in ./.venv (vedi poetry.toml)
cp .env.example .env             # poi inserisci i segreti
```

| Variabile `.env` | Descrizione |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | endpoint della risorsa Foundry, es. `https://<risorsa>.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_API_KEY` | chiave della risorsa (vale per tutti i deployment, incluso `gpt-5.4-mini`) |
| `AZURE_OPENAI_API_VERSION` | `2025-04-01-preview` (Responses API) |
| `AZURE_OPENAI_DEPLOYMENT` | deployment di default, es. `gpt-5.4` |
| `TYPESAFE_API_KEY` | chiave Jev da [console.typesafe.ai/keys](https://console.typesafe.ai/keys) |
| `JEV_MODEL` | versione pinnata di Jev, es. `jev-1.14.0` |
| `TYPESAFE_BASE_URL` | opzionale, gateway compatibile (es. OpenRouter `https://openrouter.ai/api`) |

`.env` e `.venv/` sono esclusi da git.

## Uso

```bash
poetry run jevbench tasks                       # elenca gli scenari
poetry run jevbench estimate                    # stima token e costi OFFLINE (nessuna chiamata)
poetry run jevbench check --systems all         # UNA richiesta per sistema: credenziali, deployment, logprobs
poetry run jevbench run --systems main          # confronto principale (riprende dagli esempi mancanti)
poetry run jevbench run --systems bias          # varianti di ordine per il bias di posizione
poetry run jevbench run --systems jev --tasks intent_routing --limit 20
poetry run jevbench report                      # metriche, grafici e reports/main/report.md
poetry run jevbench classify "svegliami domani alle sette" --task intent_routing_it --systems jev,gpt-5.4-mini-none
poetry run streamlit run app/streamlit_app.py   # demo interattiva
```

Preset di sistemi: `main` (i cinque sistemi del confronto), `bias` (varianti `@reversed`/`@shuffled`
di `jev` e `gpt-5.4-none`), `all`. Gli scenari si scelgono con `--tasks all`, `choice` o un elenco di
nomi. Le varianti di ordine e `jev-perm` saltano automaticamente S3, che è una scala ordinale.

### Demo

La demo Streamlit permette di:

- scegliere uno scenario e un esempio del benchmark (con la risposta attesa) o scrivere un input libero;
- vedere affiancate le decisioni di Jev, `jev-perm`, GPT-5.4 mini e GPT-5.4, con la distribuzione di
  probabilità, la latenza, il costo per 1.000 richieste e i token;
- **riordinare le opzioni** (canonico, inverso, casuale) per vedere dal vivo il bias di posizione;
- simulare la **cascata** Jev → LLM con una soglia τ regolabile;
- consultare i grafici e le tabelle del benchmark.

## Struttura

```
├── src/jev_benchmark/
│   ├── tasks/          # i 4 scenari: opzioni, descrizioni, ordini delle opzioni, campionamento
│   ├── systems/        # jev.py (typesafe-sdk, jev-perm) e foundry.py (Azure OpenAI Responses API)
│   ├── runner.py       # esecuzione asincrona, retry, warm-up, output JSONL riprendibile
│   ├── metrics.py      # qualità + IC, McNemar, calibrazione, latenza, costo, cascata
│   ├── bias.py         # bias di posizione: accuratezza per ordine, coerenza, sovra-selezione
│   ├── estimate.py     # stima offline di token e costi
│   ├── report.py       # report Markdown e grafici (tema chiaro e scuro)
│   └── cli.py          # comando `jevbench`
├── app/streamlit_app.py
├── data/benchmark/     # 4 × 500 esempi campionati (versionati) + README con licenze
├── results/<run_id>/   # predizioni per esempio + manifest (versioni, prezzi, commit)
├── reports/<run_id>/   # report.md, summary.json, figures/
├── docs/               # documentazione (vedi sotto)
└── tests/              # test offline, inclusa l'integrazione Jev con trasporto HTTP simulato
```

## Documentazione

1. [Jev di TypeSafe AI: cos'è e come si usa](docs/01-jev-typesafe.md): API, SDK, primitive, confidenza, pattern e limiti
2. [Integrazione con Microsoft Foundry](docs/02-integrazione-foundry.md): baseline GPT-5.4 e mini, architetture Jev + Foundry, mitigazione del bias
3. [Metodologia del benchmark](docs/03-metodologia.md): dataset, parità delle condizioni, metriche, bias di posizione, stima dei costi, limiti
4. [Risultati e raccomandazioni](docs/04-risultati-e-raccomandazioni.md): quando usare Jev, l'LLM o entrambi

## Test

```bash
poetry run pytest          # offline: nessuna chiamata agli endpoint dei modelli
poetry run ruff check src tests app
```

## Licenze dei dati

Banking77 (CC BY 4.0), ABCD (MIT), HelpSteer2 (CC BY 4.0) e MASSIVE (CC BY 4.0): dettagli e citazioni
in [data/README.md](data/README.md).
