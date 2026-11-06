# 04 · Risultati e raccomandazioni

> **Stato: preliminare.** I baseline GPT-5.4 sono stati misurati su tutti i 908 esempi (run `main`,
> 6 novembre 2026). Le predizioni di Jev sono in attesa della `TYPESAFE_API_KEY`. Questo documento
> riporta (1) ciò che è già misurato e (2) i criteri decisionali, fissati **prima** di vedere i
> risultati di Jev, con cui verrà formulata la raccomandazione finale. Numeri completi:
> [reports/main/report.md](../reports/main/report.md).

## 1. Cosa sappiamo già: i baseline GPT-5.4 su Foundry

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
   è servito dalla cache ($0,25/M invece di $2,50/M), perché istruzioni e 77 opzioni sono un prefisso
   costante. Il costo scende da $4,81 a $1,19 per 1.000 richieste, circa 4 volte meno. In S3 il prefisso costante è sotto la soglia di 1.024 token e la cache non si
   attiva.

## 2. Criteri decisionali (fissati prima dei risultati di Jev)

Per ogni scenario, confrontando Jev con GPT-5.4 `none` (il baseline più comparabile per latenza) e con
`medium` (il baseline di qualità):

| Esito | Condizione | Raccomandazione |
|---|---|---|
| **Jev da solo** | accuratezza di Jev non significativamente inferiore all'LLM (McNemar p ≥ 0,05 e differenza ≤ 2 punti; per S3 QWK ≥ QWK_LLM − 0,05) | sostituire l'LLM: stessa qualità con un costo e una latenza inferiori |
| **Combinazione (cascata)** | Jev significativamente peggiore, ma la cascata raggiunge la parità (o -2 punti) inoltrando **≤ 50%** delle richieste, con soglia scelta in cross-fitting | Jev su tutto il traffico e LLM solo sotto soglia: si risparmia sulla quota non inoltrata |
| **LLM** | la parità richiede di inoltrare > 70% delle richieste, oppure la probabilità di Jev non separa giusto/sbagliato (AUROC < 0,70) | tenere l'LLM. Jev resta eventualmente utile come pre-filtro economico (*noul screen*) |
| Zona grigia | inoltro tra 50% e 70% | decidere in base al vincolo dominante (latenza sincrona vs costo vs accuratezza) |

Elementi aggiuntivi che pesano sulla scelta, oltre ai numeri:

- **Volumi:** il risparmio assoluto conta solo ad alto volume. A 1 milione di richieste al mese la
  differenza tra $1-8 e pochi centesimi per 1.000 richieste vale migliaia di dollari l'anno; a 10.000
  richieste al mese è irrilevante.
- **Latenza:** in un flusso sincrono (chat, routing in tempo reale) contano p95 e p99, non la media.
- **Governance:** Jev è un servizio esterno ad Azure, in *early access* (residenza dei dati, DPA,
  SLA). La cascata include già il fallback sull'LLM in caso di indisponibilità.

## 3. Raccomandazione finale

*Da completare dopo `jevbench run --systems jev` e `jevbench report`, applicando i criteri della
sezione 2 scenario per scenario.*
