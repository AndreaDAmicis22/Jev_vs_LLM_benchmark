# 04 · Risultati e raccomandazioni

> **Stato: preliminare.** I baseline GPT-5.4 sono stati misurati sul campione **v1** (908 esempi, run
> `main`, 6 novembre 2026). Il benchmark è stato poi esteso alla **v2** (500 esempi per scenario,
> scenario italiano S4, GPT-5.4 mini, test del bias di posizione): la v2 contiene integralmente la v1,
> quindi questi risultati restano validi e verranno completati, non rifatti. Le predizioni di Jev sono
> in attesa della `TYPESAFE_API_KEY`. Questo documento riporta (1) ciò che è già misurato e (2) i criteri
> decisionali, fissati **prima** di vedere i risultati di Jev, con cui verrà formulata la
> raccomandazione finale. Numeri completi: [reports/main/report.md](../reports/main/report.md).

## 1. Cosa sappiamo già: i baseline GPT-5.4 su Foundry (campione v1)

| Scenario | Configurazione | Accuratezza [IC 95%] | Latenza p50 / p95 | Costo / 1.000 richieste |
|---|---|---|---|---|
| S1 · 77 intent | reasoning none | 87,3% [83,4-90,9] | 1,16 s / 2,46 s | $1,19 |
| | reasoning medium | 88,3% [84,7-91,6] | 1,52 s / 2,82 s | $2,14 |
| S2 · 30 azioni | reasoning none | 77,3% [72,3-82,0] | 1,21 s / 1,89 s | $3,27 |
| | reasoning medium | 80,0% [75,3-84,7] | 2,27 s / **19,2 s** | $7,69 |
| S3 · voto 0-4 | reasoning none | 39,7% esatta · QWK 0,567 · gate ≥3 72,5% | 1,15 s / 1,89 s | $2,02 |
| | reasoning medium | 45,0% esatta · QWK 0,620 · gate ≥3 74,9% | 2,80 s / 4,98 s | $4,69 |

### Osservazioni

1. **Il ragionamento aggiunge poco a questi compiti.** `medium` guadagna da 1 a 5 punti (S1 +1,0,
   S2 +2,7, S3 +5,3 esatta / +0,05 QWK) al costo di una spesa da 1,8 a 2,4 volte maggiore e di una
   latenza mediana da 1,3 a 2,4 volte maggiore. In S1 e S2 la differenza non è statisticamente
   significativa (McNemar p = 0,51 e 0,12); in S3 sì (p = 0,03).
2. **Il ragionamento crea una coda di latenza pesante.** In S2 il 19% delle richieste `medium` supera
   i 400 token di ragionamento. Le più lente (fino a ~4.100 token, 20-70 s) sono **in gran parte
   sbagliate** (10 delle 12 più lente): sui casi ambigui il modello ragiona a lungo senza risolverli.
   Per un workflow sincrono un p95 di 19 s è un problema operativo prima ancora che di costo.
3. **La confidenza dell'LLM è poco utilizzabile.** Con `none` l'LLM assegna p ≥ 0,99 al 95% (S1), 86%
   (S2) e 59% (S3) delle risposte, ma su quelle risposte sbaglia rispettivamente il 10%, il 17% e il
   55%. L'AUROC giusto/sbagliato è 0,66 / 0,77 / 0,57: la probabilità dei token distingue poco le
   risposte corrette da quelle errate. In S3 l'ECE è 0,52, cioè l'LLM è molto più sicuro di quanto sia
   accurato. Per instradare i casi incerti serve quindi un segnale migliore: è l'ipotesi chiave da
   verificare su Jev.
4. **Come giudice (S3) GPT-5.4 è più severo degli annotatori umani.** Il livello medio previsto è
   1,74-1,76 contro 2,00 nel campione. Le risposte "perfette" (4) vengono riconosciute raramente (12-15 su 60) e
   il modello tende a comprimere i voti verso 2. Come quality gate binario (≥ 3) concorda con gli umani
   nel 72-75% dei casi.
5. **Il content filter di Azure blocca l'1,7% delle valutazioni S3** (5 richieste su 300, le stesse con
   entrambe le configurazioni, errore 400). Un giudice LLM su Azure non può valutare proprio i
   contenuti problematici, che spesso sono quelli che un quality gate deve intercettare.
6. **Il prompt caching abbatte il costo quando le opzioni sono molte.** In S1 l'89% dei token di input
   è servito dalla cache ($0,25/M invece di $2,50/M), perché istruzioni, 77 opzioni e schema JSON sono
   un prefisso costante. Il costo scende da $4,81 a $1,19 per 1.000 richieste, circa 4 volte meno. In
   S3 il prefisso costante è sotto la soglia di 1.024 token e la cache non si attiva.

## 2. Criteri decisionali (fissati prima dei risultati di Jev)

> Revisione del 13 novembre 2026, ancora prima di qualsiasi risultato di Jev: il termine di paragone
> principale diventa **GPT-5.4 mini**, l'alternativa più economica che funziona. Sono aggiunti anche
> i criteri sul bias di posizione.

Per ogni scenario si confronta Jev (o `jev-perm`, se la variante migliora in modo significativo) con:

- **GPT-5.4 mini · none** → *decide la raccomandazione*: è l'alternativa che un cliente userebbe
  davvero per decisioni ripetitive ad alto volume;
- **GPT-5.4 · none** → riferimento di qualità e latenza;
- **GPT-5.4 · medium** → tetto di qualità, solo per contesto.

| Esito | Condizione (rispetto a GPT-5.4 mini) | Raccomandazione |
|---|---|---|
| **Jev da solo** | accuratezza di Jev non significativamente inferiore (McNemar p ≥ 0,05 e differenza ≤ 2 punti; per S3 QWK ≥ QWK_mini − 0,05) | sostituire l'LLM: stessa qualità con un costo e una latenza inferiori |
| **Combinazione (cascata)** | Jev significativamente peggiore, ma la cascata Jev → mini raggiunge la parità (o -2 punti) inoltrando **≤ 50%** delle richieste, con soglia scelta in cross-fitting | Jev su tutto il traffico e LLM solo sotto soglia: si risparmia sulla quota non inoltrata |
| **LLM** | la parità richiede di inoltrare > 70% delle richieste, oppure la probabilità di Jev non separa giusto/sbagliato (AUROC < 0,70) | tenere l'LLM. Jev resta eventualmente utile come pre-filtro economico (*noul screen*) |
| Zona grigia | inoltro tra 50% e 70% | decidere in base al vincolo dominante (latenza sincrona vs costo vs accuratezza) |

**Criteri sul bias di posizione** (scenari Choice):

| Esito | Condizione | Conseguenza |
|---|---|---|
| Bias trascurabile | variazione di accuratezza tra ordini ≤ 2 punti e non significativa, e coerenza nei tre ordini ≥ 90% | Jev con ordine singolo |
| Bias assorbito dalla cascata | gli esempi che cambiano risposta hanno in maggioranza probabilità sotto la soglia scelta | Jev con ordine singolo in cascata: la soglia intercetta già i casi instabili |
| Bias da mitigare | variazione > 2 punti o significativa, oppure cambi con probabilità alta | usare `jev-perm` (media su 3 ordini) se recupera accuratezza; il costo in token di Jev resta comunque trascurabile |
| Scenario italiano (S4) | si applicano gli stessi criteri della tabella principale | se Jev perde in S4 più che in S1, va segnalata una dipendenza dalla lingua da validare sui dati del cliente |

Elementi aggiuntivi che pesano sulla scelta, oltre ai numeri:

- **Volumi:** il risparmio assoluto conta solo ad alto volume. A 1 milione di richieste al mese la
  differenza tra $1-8 e pochi centesimi per 1.000 richieste vale migliaia di dollari l'anno; a 10.000
  richieste al mese è irrilevante.
- **Latenza:** in un flusso sincrono (chat, routing in tempo reale) contano p95 e p99, non la media.
- **Governance:** Jev è un servizio esterno ad Azure, in *early access* (residenza dei dati, DPA,
  SLA). La cascata include già il fallback sull'LLM in caso di indisponibilità.

## 3. Raccomandazione finale

*Da completare dopo i run della v2 e `jevbench report`, applicando i criteri della sezione 2 scenario
per scenario:*

```bash
poetry run jevbench estimate               # costo previsto, offline
poetry run jevbench check --systems all    # una richiesta per sistema
poetry run jevbench run --systems main     # Jev, jev-perm, GPT-5.4 mini + esempi nuovi di GPT-5.4
poetry run jevbench run --systems bias     # varianti di ordine per il bias di posizione
poetry run jevbench report
```
