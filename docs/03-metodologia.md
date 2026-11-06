# 03 · Metodologia del benchmark

## Domanda

Per decisioni **ricorrenti tra opzioni predefinite**, Jev può sostituire o affiancare un LLM
generativo? Il confronto riguarda **qualità**, **latenza** e **costo**, più un quarto aspetto che
determina se una combinazione è possibile: l'**affidabilità della confidenza**, cioè quanto ciascun
sistema "sa di sapere".

## Scenari e dataset

Tre scenari, tutti riconducibili a una scelta tra opzioni, su dataset pubblici con etichette umane.

| | S1 · Richieste in ingresso | S2 · Passo successivo in un workflow | S3 · Qualità di un output generato |
|---|---|---|---|
| Dataset | Banking77, split *test* | ABCD v1.1, split *test* | HelpSteer2, split *validation* |
| Input (`state`) | messaggio del cliente | SOP del workflow corrente + conversazione fino al momento della decisione | prompt dell'utente + risposta generata da un LLM |
| Decisione | 1 di **77** intent | 1 di **30** azioni di sistema | livello di *helpfulness* **0-4** (ordinale) |
| Gold | etichetta del dataset | bottone effettivamente cliccato dall'agente umano | voto degli annotatori umani |
| Campione | 308 = 4 × 77 | 300 = 10 × 30, al massimo un punto di decisione per conversazione | 300 = 60 × 5 |
| Primitive Jev | `Choice` | `Choice` | `Score` (5 livelli) |

**Campionamento bilanciato e stratificato** (seed 42, `jevbench prepare`): ogni classe pesa allo stesso
modo, quindi le metriche macro sono significative e nessuna classe frequente domina. Il prezzo è che
l'accuratezza **non** rispecchia una distribuzione di produzione: in produzione le classi frequenti
pesano di più.

Scelte specifiche:

- **S1:** le descrizioni dei 77 intent sono state scritte guardando **solo lo split train**, mai il
  test. In più seguono la semantica reale delle etichette: per esempio `get_physical_card`, nel
  dataset, riguarda soprattutto il PIN della carta fisica.
- **S2:** è lo scenario "workflow engine": il workflow è noto (in un sistema reale lo sceglierebbe il
  router di S1) e la decisione riguarda **a che punto della procedura** ci si trova e quale ramo si
  applica, per esempio "Oracle = no → controlla la membership". La SOP del workflow, ricavata dalle
  linee guida ufficiali ABCD, fa parte dello stato. Le descrizioni delle azioni iniziano con il nome
  del bottone usato nella SOP (`[Pull up Account]`…) per collegare le due cose.
- **S3:** la rubrica dei 5 livelli riprende le linee guida di annotazione di HelpSteer2. Oltre al voto
  esatto si valuta un **quality gate** binario (accetta se helpfulness ≥ 3), che è il caso d'uso tipico
  in produzione.

## Parità delle condizioni

| Aspetto | Come è garantito |
|---|---|
| Stessa domanda, stesse opzioni, stesse descrizioni | un unico oggetto `Task` produce sia la domanda Jev (`instructions` + `criteria`) sia il system prompt e lo schema JSON dell'LLM |
| Stesso input | gli stessi campi di `state`: a Jev come oggetto JSON, all'LLM come sezioni `<campo>…</campo>` |
| Nessun vantaggio di prompt engineering | zero-shot per entrambi, senza esempi né ottimizzazioni specifiche per un sistema |
| Solo risposte valide | Jev sceglie per costruzione tra le opzioni; l'LLM è vincolato da Structured Outputs (`enum` strict) |
| Stesso ordine delle opzioni | ordine canonico del dataset per entrambi (Jev tende a favorire la prima opzione) |
| Stesse condizioni di rete | stessa macchina, stessa finestra temporale, concorrenza 8 per entrambi, una richiesta di warm-up non misurata per aprire le connessioni |

## Sistemi confrontati

| Sistema | Configurazione |
|---|---|
| **Jev** | `jev-1.14.0` (versione pinnata), SDK `typesafe-sdk` 0.7.5, `AsyncTypeSafeClient`, retry dello SDK disattivati (gestiti dal runner) |
| **GPT-5.4 · reasoning none** | deployment Foundry `gpt-5.4` → `gpt-5.4-2026-03-05`, East US, Responses API, `reasoning.effort="none"`, logprobs attivi, `store=False` |
| **GPT-5.4 · reasoning medium** | come sopra con `reasoning.effort="medium"` (senza logprobs, non disponibili con il ragionamento) |

## Misure

### Qualità

- **Accuratezza**: quota di decisioni uguali al gold. Le richieste fallite contano come errori.
- **Macro-F1** (S1, S2): media non pesata dell'F1 per classe.
- **S3, ordinale**: accuratezza esatta, accuratezza entro ±1 livello, MAE, **QWK** (*quadratic weighted
  kappa*, la metrica standard di accordo su scale ordinali), Spearman tra il punteggio atteso e il voto
  umano, e accuratezza/F1 del gate ≥ 3. La predizione discreta di Jev è il livello **più probabile**;
  per Spearman si usa il punteggio atteso (media pesata dei livelli). Le metriche diverse
  dall'accuratezza esatta sono calcolate sulle richieste con risposta.
- **Incertezza statistica**: intervalli di confidenza al 95% con bootstrap percentile (2.000
  ricampionamenti). I confronti appaiati tra sistemi usano il **test di McNemar esatto**. Con circa
  300 esempi l'IC dell'accuratezza è di circa ±4-5 punti.

### Latenza

- Tempo **end-to-end lato client** (`perf_counter` attorno alla chiamata HTTP) del tentativo andato a
  buon fine: è ciò che vede un'applicazione. Si riportano media, p50, p90, p95 e p99.
- Per Jev si registra anche il **tempo lato server** dichiarato dal gateway
  (`x-envoy-upstream-service-time`); Azure non espone un header equivalente.
- Contesto di rete: client in Italia; Foundry in **East US**; TypeSafe dietro Cloudflare, edge di
  Milano (`CF-RAY …-MXP`), con origine non dichiarata. La latenza assoluta dipende da dove gira
  l'applicazione: da una VM nella stessa regione del servizio i tempi di rete si riducono.

### Costo

- Calcolato **per richiesta** dai token riportati dalle API e dai listini pubblici (novembre 2026):
  Jev $0,042/M token input, output gratuito; GPT-5.4 $2,50/M input, $0,25/M input in cache, $15/M
  output, con i token di ragionamento fatturati come output.
- Il **prompt caching** di Azure è incluso così come avviene, perché in produzione avverrebbe allo
  stesso modo. I token di Jev e di GPT-5.4 sono contati da tokenizer diversi.

### Confidenza e calibrazione

- **Probabilità della risposta scelta**: per Jev `probabilities[choice]` (S1, S2) o la probabilità del
  livello scelto (S3); per GPT-5.4 `none` la probabilità della sequenza di token della label, dai
  logprobs.
- **ECE** (*expected calibration error*, 10 fasce), **Brier** e **AUROC giusto/sbagliato**: quanto la
  probabilità separa le risposte corrette da quelle errate, cioè quanto è utile per decidere cosa
  inoltrare.
- **Curve accuratezza/copertura**: accuratezza ottenuta accettando solo la quota di risposte più
  sicure. Sono *tie-aware*: risposte con la stessa probabilità non sono separabili da nessuna soglia.

## Combinazione: la cascata Jev → LLM

```
risposta = Jev            se segnale_Jev ≥ τ
           LLM            altrimenti (la richiesta paga Jev + LLM, in sequenza)
```

- **Segnale** come da guida TypeSafe: probabilità dell'opzione scelta per `Choice` (S1, S2),
  `confidence` per `Score` (S3).
- **Simulazione offline**: entrambi i sistemi hanno risposto a tutti gli esempi, quindi per ogni soglia
  τ si combinano le risposte, le latenze e i costi **misurati sullo stesso esempio**. Non servono altre
  chiamate.
- **Punto operativo senza overfitting**: la soglia si sceglie con *2-fold cross-fitting* (si tara τ
  su metà degli esempi e si misura sull'altra metà, poi si invertono le metà). Si riportano due
  obiettivi: la **parità** con l'accuratezza dell'LLM da solo e una tolleranza di **-2 punti**.
- **Varianti non simulate**: chiamata parallela (latenza minore, costo pieno dell'LLM) e cascata a tre
  livelli (Jev → `none` → `medium`).

## Minacce alla validità

| Limite | Effetto | Mitigazione |
|---|---|---|
| Rumore nelle etichette (Banking77 ha etichette ambigue note; in ABCD gli agenti non sempre seguono la SOP e alcuni workflow ammettono le azioni "in qualunque ordine"; HelpSteer2 è soggettivo) | il tetto massimo di accuratezza è < 100% e si abbassa per tutti | confronto **relativo** e appaiato sugli stessi esempi |
| Campioni di ~300 esempi | IC di circa ±5 punti | IC bootstrap e McNemar riportati sempre |
| Un solo run | l'LLM non è perfettamente deterministico | i risultati grezzi per esempio sono versionati e si possono ripetere i run |
| Prompt non ottimizzati | entrambi i sistemi potrebbero migliorare con descrizioni tarate | parità garantita dalla definizione condivisa |
| Bias di Jev verso la prima opzione | possibile vantaggio o svantaggio sistematico | ordine identico per entrambi; test di riordino suggerito come estensione |
| Dataset in inglese | non misura le prestazioni su testi in italiano | da validare sui dati del cliente |
| Content filter di Azure | alcune richieste S3 vengono rifiutate (contano come errori) | riportato separatamente |
| Prezzi e versioni di novembre 2026 | i rapporti di costo possono cambiare | prezzi parametrizzati in [`config.py`](../src/jev_benchmark/config.py) |

## Riproducibilità

```bash
poetry install --with demo,dev        # crea ./.venv
cp .env.example .env                  # e compila i segreti
poetry run jevbench prepare           # (opzionale) rigenera data/benchmark/ dai dataset pubblici
poetry run jevbench check             # verifica le credenziali di Jev e Foundry
poetry run jevbench run --run-id main # riprende da dove si era interrotto
poetry run jevbench report --run-id main
```

I risultati per esempio (`results/<run_id>/<scenario>/<sistema>.jsonl`), il manifest con versioni,
prezzi e commit, e il report (`reports/<run_id>/`) sono versionati nel repository.
