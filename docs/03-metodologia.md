# 03 · Metodologia del benchmark

## Domanda

Per decisioni **ricorrenti tra opzioni predefinite**, Jev può sostituire o affiancare un LLM
generativo? Il confronto riguarda **qualità**, **latenza** e **costo**, più un quarto aspetto che
determina se una combinazione è possibile: l'**affidabilità della confidenza**, cioè quanto ciascun
sistema "sa di sapere".

## Scenari e dataset

Quattro scenari, tutti riconducibili a una scelta tra opzioni, su dataset pubblici con etichette umane.

| | S1 · Richieste in ingresso | S2 · Passo successivo in un workflow | S3 · Qualità di un output generato | S4 · Richieste in italiano |
|---|---|---|---|---|
| Dataset | Banking77, split *test* | ABCD v1.1, split *test* | HelpSteer2, split *validation* | MASSIVE 1.1 it-IT, split *test* (+ *dev* per gli intent rari) |
| Input (`state`) | messaggio del cliente | SOP del workflow corrente + conversazione fino al momento della decisione | prompt dell'utente + risposta generata da un LLM | richiesta a un assistente virtuale, in italiano |
| Decisione | 1 di **77** intent | 1 di **30** azioni di sistema | livello di *helpfulness* **0-4** (ordinale) | 1 di **60** intent |
| Lingua di istruzioni e opzioni | inglese | inglese | inglese | **italiano** |
| Gold | etichetta del dataset | bottone effettivamente cliccato dall'agente umano | voto degli annotatori umani | etichetta del dataset |
| Campione (v2) | 500, 6-7 per intent | 500, 16-17 per azione, al massimo un punto di decisione per conversazione | 500: 78 + 93 (tutti i livelli 0 e 1) + ~110 per i livelli 2-4 | 500, fino a 9 per intent |
| Primitive Jev | `Choice` | `Choice` | `Score` (5 livelli) | `Choice` |

**Campionamento stratificato** (seed 42, `jevbench prepare`): le quote sono distribuite in modo
uniforme tra le classi e le classi con pochi esempi vengono prese per intero. Ogni classe pesa allo
stesso modo, quindi le metriche macro sono significative e nessuna classe frequente domina. Il prezzo
è che l'accuratezza **non** rispecchia una distribuzione di produzione, dove le classi frequenti
pesano di più.

**500 esempi per scenario e continuità con la v1.** La prima versione usava 308 / 300 / 300 esempi
(IC di circa ±5 punti). Con 500 esempi l'IC dell'accuratezza scende a circa ±3-4 punti. Il
campionamento è costruito in modo che la v2 **contenga integralmente** la v1:
- per S1 e S3 ogni classe viene mescolata una sola volta con il seed e se ne prendono i primi *k*;
- per S2 la prima fase riproduce esattamente il campionamento v1 e la seconda lo estende.

I risultati già misurati restano quindi validi e si eseguono solo gli esempi nuovi. Un test
automatico verifica che ogni risultato salvato corrisponda a un esempio identico del benchmark.

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
- **S4:** le frasi di MASSIVE sono state **localizzate da traduttori professionisti madrelingua**
  partendo dal corpus inglese SLURP: non sono traduzioni automatiche. Usare testi italiani di questo
  tipo, invece di tradurre Banking77 con un LLM, evita di favorire GPT-5.4. Istruzioni e descrizioni
  dei 60 intent sono in italiano, come lo sarebbero in un'applicazione italiana, e sono state scritte
  dallo split train. Il dominio è un assistente virtuale (sveglie, musica, domotica, meteo…), più
  generico del customer service; a differenza di S1 lo scenario misura insieme due effetti, input
  italiano e istruzioni italiane, come in un deployment reale.

## Parità delle condizioni

| Aspetto | Come è garantito |
|---|---|
| Stessa domanda, stesse opzioni, stesse descrizioni | un unico oggetto `Task` produce sia la domanda Jev (`instructions` + `criteria`) sia il system prompt e lo schema JSON dell'LLM |
| Stesso input | gli stessi campi di `state`: a Jev come oggetto JSON, all'LLM come sezioni `<campo>…</campo>` |
| Nessun vantaggio di prompt engineering | zero-shot per entrambi, senza esempi né ottimizzazioni specifiche per un sistema |
| Solo risposte valide | Jev sceglie per costruzione tra le opzioni; l'LLM è vincolato da Structured Outputs (`enum` strict) |
| Stesso ordine delle opzioni | ordine canonico del dataset per entrambi nel confronto principale; l'effetto dell'ordine è misurato a parte (vedi *Bias di posizione*) |
| Stesse condizioni di rete | stessa macchina, stessa finestra temporale, concorrenza 8 per entrambi, una richiesta di warm-up non misurata per aprire le connessioni |

## Sistemi confrontati

| Sistema (nome nella CLI) | Ruolo | Configurazione |
|---|---|---|
| **Jev** (`jev`) | sistema in valutazione | `jev-1.14.0` (versione pinnata), SDK `typesafe-sdk` 0.7.5, `AsyncTypeSafeClient`, retry dello SDK disattivati (gestiti dal runner) |
| **Jev · media su 3 ordini** (`jev-perm`) | mitigazione del bias di posizione | stessa Choice ripetuta 3 volte **nella stessa richiesta** (ordine canonico, inverso, casuale) con media delle probabilità nel codice; il disaccordo tra i 3 ordini è un segnale di incertezza in più. Solo scenari Choice |
| **GPT-5.4 mini · reasoning none** (`gpt-5.4-mini-none`) | **alternativa economica**, il vero termine di paragone sul costo | deployment Foundry `gpt-5.4-mini`, Responses API, `reasoning.effort="none"`; logprobs richiesti con fallback automatico se il deployment li rifiuta |
| **GPT-5.4 · reasoning none** (`gpt-5.4-none`) | riferimento di qualità e latenza | deployment Foundry `gpt-5.4` → `gpt-5.4-2026-03-05`, East US, `reasoning.effort="none"`, logprobs attivi, `store=False` |
| **GPT-5.4 · reasoning medium** (`gpt-5.4-medium`) | tetto di qualità | come sopra con `reasoning.effort="medium"` (senza logprobs, non disponibili con il ragionamento) |

Perché GPT-5.4 mini. Confrontare Jev solo con il modello di punta sovrastimerebbe il suo vantaggio di
costo, perché per classificare milioni di richieste ripetitive nessuno userebbe GPT-5.4. La domanda
giusta è: *Jev conviene rispetto all'alternativa più economica che funziona?* Mini costa circa il 30%
di GPT-5.4 ($0,75/M input, $0,075/M in cache, $4,50/M output), resta dentro Azure con le stesse API e
lo stesso prompt caching, ed è abbastanza capace da essere un avversario credibile anche su S2 e S3.

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
  ricampionamenti). I confronti appaiati tra sistemi usano il **test di McNemar esatto**. Con 500
  esempi l'IC dell'accuratezza è di circa ±3-4 punti (era ±4-5 con i ~300 esempi della v1).

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
  output; GPT-5.4 mini $0,75/M input, $0,075/M in cache, $4,50/M output. I token di ragionamento sono
  fatturati come output. Un deployment senza prezzo configurato blocca il run: i costi non vengono mai
  stimati a caso.
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

- **Segnale** come da guida TypeSafe: probabilità dell'opzione scelta per `Choice` (S1, S2, S4),
  `confidence` per `Score` (S3). Per `jev-perm` anche il disaccordo tra i tre ordini fa inoltrare la
  richiesta.
- **Cascate calcolate**: `jev` e `jev-perm` come primo stadio, ciascuna verso GPT-5.4 mini, GPT-5.4
  none e GPT-5.4 medium.
- **Simulazione offline**: entrambi i sistemi hanno risposto a tutti gli esempi, quindi per ogni soglia
  τ si combinano le risposte, le latenze e i costi **misurati sullo stesso esempio**. Non servono altre
  chiamate.
- **Punto operativo senza overfitting**: la soglia si sceglie con *2-fold cross-fitting* (si tara τ
  su metà degli esempi e si misura sull'altra metà, poi si invertono le metà). Si riportano due
  obiettivi: la **parità** con l'accuratezza dell'LLM da solo e una tolleranza di **-2 punti**.
- **Varianti non simulate**: chiamata parallela (latenza minore, costo pieno dell'LLM) e cascata a tre
  livelli (Jev → `none` → `medium`).

## Bias di posizione: l'ordine delle opzioni cambia la decisione?

TypeSafe documenta che Jev "tende a favorire la prima opzione" di una Choice
([model jaggedness](https://docs.typesafe.ai/model-jaggedness.md), *Choice option order*) ma non
quantifica l'effetto. Il fenomeno è noto anche negli LLM nei test a scelta multipla (Zheng et al.,
ICLR 2024; Pezeshkpour e Hruschka, 2023).

**Protocollo.** Negli scenari Choice (S1, S2, S4) **Jev** e **GPT-5.4 none** vengono rieseguiti sugli
stessi 500 esempi con tre ordini delle opzioni:
- **canonico**: il run principale;
- **inverso** (`@reversed`);
- **casuale** (`@shuffled`): una permutazione diversa per ogni esempio, generata da un seed fisso e
  dall'id dell'esempio, quindi riproducibile.

S3 è escluso perché lì l'ordine coincide con la scala 0-4. Con l'ordine casuale la posizione di
un'opzione è indipendente dalla sua identità: qualunque eccesso di scelte in cima alla lista è un puro
effetto di posizione.

**Misure** (modulo [`bias.py`](../src/jev_benchmark/bias.py), sezione dedicata del report):

| Misura | Domanda a cui risponde |
|---|---|
| Accuratezza per ordine, con IC e McNemar rispetto al canonico | il risultato cambia solo riordinando le opzioni? |
| Coerenza: quota di esempi con la stessa risposta nei tre ordini | quante decisioni sono instabili? |
| Scelte in 1ª posizione (e nelle prime 10% delle posizioni) rispetto alle risposte corrette che si trovano lì | c'è sovra-selezione della cima della lista? |
| Rapporto scelte/corrette per decile della posizione (ordine casuale) e, per Jev, la massa di probabilità per decile | dove si concentra lo spostamento? |
| Probabilità (ordine canonico) degli esempi che cambiano risposta, quota sotto 0,5, AUROC | la soglia della cascata intercetta già i casi instabili? |

**Mitigazione misurata: `jev-perm`.** È il rimedio indicato da TypeSafe ("riordinare le opzioni e
verificare la coerenza"), applicabile in produzione. Nella stessa richiesta si inserisce la stessa
domanda tre volte, con id diversi e le opzioni in ordine canonico, inverso e casuale, poi nel codice si
fa la media delle probabilità. Le domande di una richiesta sono valutate in parallelo e in isolamento:
la latenza cambia poco e si pagano solo i token delle opzioni ripetute, circa il triplo dell'input in
S1. Il confronto con `jev` mostra quanto accuratezza e calibrazione recuperano; il disaccordo tra gli
ordini diventa un segnale di inoltro nella cascata.

## Stima dei costi prima di eseguire

`jevbench estimate` calcola **offline** token e costo di un run pianificato, senza chiamare alcun
endpoint ([`estimate.py`](../src/jev_benchmark/estimate.py)):

- costruisce gli stessi prompt che verrebbero inviati e ne conta i token con `o200k_base`, il
  tokenizer della famiglia GPT-5. Per Jev è un'approssimazione, perché il suo tokenizer non è pubblico;
- include lo schema JSON di Structured Outputs, che Azure fattura come input, e modella la prompt
  cache come la applica Azure: prefisso costante di almeno 1.024 token, a blocchi di 128. Con l'ordine
  casuale la cache non si applica;
- dove esistono risultati misurati sullo stesso scenario, calibra token di input, quota di cache e
  token di output sui dati reali;
- esclude gli esempi già eseguiti, come fa `jevbench run`.

Sui run della v1 il conteggio dei token di input prima della calibrazione è entro +4/+6% dai valori
misurati, quindi leggermente prudente.

## Minacce alla validità

| Limite | Effetto | Mitigazione |
|---|---|---|
| Rumore nelle etichette (Banking77 ha etichette ambigue note; in ABCD gli agenti non sempre seguono la SOP e alcuni workflow ammettono le azioni "in qualunque ordine"; HelpSteer2 è soggettivo) | il tetto massimo di accuratezza è < 100% e si abbassa per tutti | confronto **relativo** e appaiato sugli stessi esempi |
| Campioni di 500 esempi | IC di circa ±3-4 punti: differenze di 2 punti restano difficili da rilevare | IC bootstrap e McNemar riportati sempre |
| Run in giorni diversi | gli esempi della v1 sono stati misurati il 6/11, i nuovi in seguito: la latenza può risentire del momento | timestamp per richiesta nei risultati; si possono rieseguire tutti gli LLM nella stessa sessione (vedi stima costi) |
| Un solo run | l'LLM non è perfettamente deterministico | i risultati grezzi per esempio sono versionati e si possono ripetere i run |
| Prompt non ottimizzati | entrambi i sistemi potrebbero migliorare con descrizioni tarate | parità garantita dalla definizione condivisa |
| Bias di posizione | l'ordine canonico può avvantaggiare o svantaggiare un sistema | misurato con tre ordini; ordine identico per tutti nel confronto principale |
| S4: un solo dataset italiano, dominio assistente virtuale | dice poco su testi italiani di dominio (reclami, ticket tecnici) | da validare sui dati del cliente |
| Content filter di Azure | alcune richieste S3 vengono rifiutate (contano come errori) | riportato separatamente |
| Prezzi e versioni di novembre 2026 | i rapporti di costo possono cambiare | prezzi parametrizzati in [`config.py`](../src/jev_benchmark/config.py) |

## Riproducibilità

```bash
poetry install --with demo,dev        # crea ./.venv
cp .env.example .env                  # e compila i segreti
poetry run jevbench prepare           # (opzionale) rigenera data/benchmark/ dai dataset pubblici
poetry run jevbench estimate          # costo previsto, offline
poetry run jevbench check --systems all   # UNA richiesta per sistema (~$0,01): credenziali, deployment, logprobs
poetry run jevbench run --systems main    # confronto principale (riprende da dove si era interrotto)
poetry run jevbench run --systems bias    # bias di posizione: varianti @reversed / @shuffled
poetry run jevbench report --run-id main
```

I risultati per esempio (`results/<run_id>/<scenario>/<sistema>.jsonl`), il manifest con versioni,
prezzi e commit, e il report (`reports/<run_id>/`) sono versionati nel repository.
