# 01 · Jev di TypeSafe AI: cos'è e come si usa

> Stato delle informazioni: novembre 2026, documentazione ufficiale su
> [docs.typesafe.ai](https://docs.typesafe.ai) e SDK `typesafe-sdk` 0.7.5. Le parti marcate
> *"nel PoC"* rimandano al codice di questo repository.

## In breve

Jev è il primo modello della famiglia **System One** di TypeSafe AI: un **modello di decisione**, non
un modello generativo. Riceve:

- uno **stato** (`state`): il contesto su cui decidere, come testo, oggetto JSON o array;
- una o più **domande tipizzate** (`questions`), ognuna con le opzioni ammesse e la loro descrizione.

Per ogni domanda restituisce una **distribuzione di probabilità** sulle opzioni. Non scrive testo e
non segue istruzioni libere: sceglie, valuta o risponde sì/no.

| Aspetto | Valore |
|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` (elenco modelli: `GET /v1/models`) |
| Autenticazione | `Authorization: Bearer <TYPESAFE_API_KEY>` (chiavi su [console.typesafe.ai/keys](https://console.typesafe.ai/keys)) |
| Modelli | `jev-1.14.0` (da 14/10/2026, aggiunge le domande `box`), `jev-1.13.0`; alias `jev-latest` e `jev-preview` (oggi entrambi → 1.14.0) |
| Prezzo | **$0,042 per milione di token di input**; i token di output sono gratuiti |
| Contesto | 32k token per domanda (stato + domanda), 64k per richiesta; fino a 262k con l'header `TypeSafe-Danger-Zone-Long-Context: true`, più lento e meno accurato |
| Input | testo (stringa, JSON, array). Immagini solo con l'header `TypeSafe-Danger-Zone-Images: true` |
| Streaming | no |
| Rate limit | per organizzazione, in token/s e richieste/s (visibili in console); `429` e `529` vanno ritentati con backoff |

> **Pin della versione.** Gli alias si spostano a ogni release e i valori delle risposte possono
> cambiare tra versioni. Per un benchmark o per soglie già tarate usare un ID versionato: *nel PoC*
> `JEV_MODEL=jev-1.14.0` in `.env`.

## Le primitive (tipi di domanda)

| Tipo | Quando usarlo | Cosa si invia | Cosa restituisce |
|---|---|---|---|
| `choice` | scegliere **una** opzione tra 1 e 255 | `criteria`: mappa `opzione → descrizione` | `choice` (opzione più probabile), `probabilities` (somma 1), `confidence` |
| `score` | valutare su una scala **ordinata** (1-10 livelli) | `criteria`: lista di descrizioni dei livelli; la posizione è il punteggio, da 0 | `score` (media pesata dei livelli, può essere frazionaria), `probabilities` per livello, `confidence`, `legend` |
| `noul` | vero/falso, sì/no | `criteria` opzionale con chiavi `true`/`false` | `noul`: probabilità del "sì" (0-1) |
| `box` | localizzare qualcosa in un'immagine | `instructions` + immagine | coordinate normalizzate + `confidence` |

Le domande di una stessa richiesta sono valutate **in parallelo e in isolamento**: aggiungere domande
costa token ma quasi nessuna latenza ("speculative fan-out").

## Esempio di chiamata HTTP

```bash
curl https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-1.14.0",
    "state": {"customer_message": "Ho ordinato la carta 10 giorni fa e non è ancora arrivata"},
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "Which intent best describes what the customer needs?",
        "criteria": {
          "card_arrival": "The ordered card has not arrived yet; the customer wants to track it.",
          "card_delivery_estimate": "How long card delivery takes, express or scheduled delivery.",
          "lost_or_stolen_card": "The card was lost or stolen."
        }
      }
    }
  }'
```

Risposta (forma documentata; i valori sono illustrativi):

```json
{
  "model": "jev-1.14.0",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "card_arrival",
      "confidence": 0.91,
      "probabilities": {"card_arrival": 0.94, "card_delivery_estimate": 0.05, "lost_or_stolen_card": 0.01}
    }
  },
  "usage": {"input_tokens": 142, "output_tokens": 0}
}
```

## SDK Python (`typesafe-sdk`)

```bash
poetry add typesafe-sdk        # oppure: pip install typesafe-sdk  (Python >= 3.10)
```

Il client legge `TYPESAFE_API_KEY` dall'ambiente; il modello si passa al client o alla singola chiamata.

```python
from typesafe_sdk import Choice, Score, TypeSafeClient

with TypeSafeClient(model="jev-1.14.0") as client:
    response = client.system_one(
        state={"prompt": "...", "response": "..."},
        questions={
            "helpfulness": Score(
                instructions="Rate the overall helpfulness of the assistant response to the user prompt.",
                criteria=["Not helpful at all ...", "Borderline unhelpful ...", "Partially helpful ...",
                          "Mostly helpful ...", "Perfectly helpful ..."],
            ),
            "language": Choice(
                instructions="In which language is the response written?",
                criteria={"italian": None, "english": None, "other": None},  # None = nome autoesplicativo
            ),
        },
    )

answer = response.answers["helpfulness"]
print(answer.score, answer.confidence, answer.probabilities)   # es. 2.7, 0.48, {0: .02, 1: .05, ...}
print(response.answers["language"].choice)
print(response.usage.input_tokens, response.model)             # token fatturati, versione effettiva
```

Altri punti utili dello SDK:

- **Async:** `AsyncTypeSafeClient` ha la stessa interfaccia (`await client.system_one(...)`), da
  chiudere con `await client.aclose()` oppure con `async with`. *Nel PoC*:
  [`systems/jev.py`](../src/jev_benchmark/systems/jev.py).
- **Retry:** di default 2 tentativi su `429/5xx/529`, nel rispetto di `retry-after`; si configura con
  `RetryPolicy(max_retries=..., timeout=...)`. *Nel PoC* sono disattivati nello SDK e gestiti dal
  runner, così la latenza misura il singolo tentativo.
- **Accessor tipizzati:** `response.choices[...]`, `response.scores[...]`, `response.nouls[...]`.
- **Metadati:** `response.request_id` e `response.raw_http_response` (header inclusi). L'header
  `x-envoy-upstream-service-time` riporta il tempo di elaborazione lato server.
- **Eccezioni:** `TypeSafeAuthenticationError`, `TypeSafeRateLimitError`, `TypeSafeBadRequestError`
  (es. `max_tokens_exceeded`), `TypeSafeUnprocessableEntityError` (validazione), `TypeSafeAPITimeoutError`…
- **Integrazioni terze:** Pydantic AI (`pydantic-ai-slim[typesafe]`, `TypeSafeModel`) e LiteLLM
  (pass-through `/typesafe/v1/systemone`, utile come gateway con chiavi virtuali e cost tracking).

## Probabilità e confidenza: come usarle

- `probabilities` è la distribuzione completa: è il segnale principale.
- `confidence` (0-1) misura quanto la distribuzione è **concentrata**, non la probabilità che la
  risposta sia giusta. Per `choice` vale `(p_max − 1/n) / (1 − 1/n)`. Per `score` penalizza anche la
  massa su livelli lontani da quello più probabile. Per `noul` la si ricava come `|2p − 1|`.
- Linee guida ufficiali:
  - per `choice`, mettere la soglia sulla **probabilità dell'opzione su cui si agisce**. A parità di
    numero di opzioni, la confidence è una trasformazione monotona di `p_max` e non aggiunge
    informazione;
  - per `score`, controllare prima la `confidence` (la risposta è "decisa"?) e poi instradare sul
    punteggio. Vicino a un confine conviene sommare le probabilità dei livelli da un lato;
  - soglie proporzionate al rischio: per esempio < 0,5 non agire, > 0,9 agire senza conferma, in
    mezzo con cautela; per `noul` si parte da 0,7 (sì) e 0,3 (no). Vanno poi tarate sui propri dati.

*Nel PoC* la cascata Jev → LLM usa esattamente questi segnali: `probabilities[choice]` per S1/S2 e
`confidence` per S3 (si veda [03-metodologia.md](03-metodologia.md)).

## Pattern consigliati da TypeSafe

| Pattern | Idea | Beneficio |
|---|---|---|
| Speculative fan-out | tutte le domande che potrebbero servire in una sola richiesta; il codice usa quelle rilevanti | costo, velocità |
| Noul screen | una domanda sì/no economica prima della fase costosa | costo, velocità |
| Companion noul | affiancare a una `choice` una `noul` "almeno un'opzione è adatta?" invece di "nessuna delle precedenti" | affidabilità |
| Confidence routing | `score` a bassa confidenza → persona o altro sistema | affidabilità, sicurezza |
| Probability routing | soglie più alte per azioni più rischiose | affidabilità, sicurezza |
| Composite scoring | più domande atomiche, pesi e combinazione nel codice | costo, affidabilità |

## Scrivere buone domande

1. **Una cosa per domanda.** Decisioni composite → più domande atomiche, combinate nel codice.
2. **Opzioni distinte e descritte.** Nome e descrizione arrivano entrambi al modello. Per opzioni
   facilmente confondibili la descrizione può essere un oggetto (`focus`, `not_for`, `examples`…: i
   nomi dei campi sono liberi).
3. **Elencare tutte le opzioni** (costano pochi token) e prevedere `other` o una companion noul quando
   la lista può non coprire tutti i casi.
4. **Stato minimo e pertinente.** Il contenuto irrilevante distrae: filtrare nel codice prima di inviare.
5. **Istruzioni dirette.** Niente doppie negazioni o catene di inferenze, ed evitare criteri in
   contraddizione con la domanda.
6. **Aritmetica, conteggi e date nel codice**, non nel modello.

## Limiti noti ("model jaggedness")

Fonte: [docs.typesafe.ai/model-jaggedness](https://docs.typesafe.ai/model-jaggedness.md).

| Punto debole | Mitigazione suggerita |
|---|---|
| Lettura letterale di negazioni, quantificatori, condizioni implicite | condizioni esplicite; domande letterali più semplici combinate nel codice |
| Matematica, conteggi, rappresentazioni numeriche | farli nel codice; una `noul` per elemento e somma nel codice |
| Date e intervalli temporali | estrarre le parti con `choice` (con opzione "non indicato") e confrontare nel codice |
| Indirezione e ragionamento multi-hop | istruzioni dirette, nominare le parti di stato rilevanti |
| Stato ampio con dettagli irrilevanti | inviare solo i campi necessari, oppure filtrare con una `noul` |
| Contenuto avversario nello stato (prompt injection) | criteri espliciti, test sui casi limite |
| Bias verso la **prima opzione** delle `choice` | riordinare le opzioni e verificare la coerenza; nel benchmark è misurato con tre ordini e mitigato con la media su più ordini (`jev-perm`), vedi [03-metodologia.md](03-metodologia.md) |
| Livelli di `score` debolmente calibrati come grandezze | usare la media solo per confronti con soglie |
| Generazione di testo | non è il suo compito: lasciarla a un LLM, Jev sceglie tra candidati |

In sintesi TypeSafe raccomanda di usare Jev per compiti "System One" (decisioni rapide e intuitive) e
di lasciare a un LLM i compiti "System Two" (ragionamento esplicito in più passi). Il benchmark misura
proprio dove passa questo confine nei tre scenari.

## Riferimenti

- Documentazione: [docs.typesafe.ai](https://docs.typesafe.ai) ·
  [API reference](https://docs.typesafe.ai/api/system-one.md) ·
  [Confidence](https://docs.typesafe.ai/confidence.md) · [Patterns](https://docs.typesafe.ai/patterns.md) ·
  [Models & pricing](https://docs.typesafe.ai/models.md) · [Errori e limiti](https://docs.typesafe.ai/api/errors.md)
- SDK: [`typesafe-sdk` su PyPI](https://pypi.org/project/typesafe-sdk/) ·
  [Pydantic AI](https://pydantic.dev/docs/ai/models/typesafe) ·
  [LiteLLM pass-through](https://docs.litellm.ai/docs/pass_through/typesafe)
