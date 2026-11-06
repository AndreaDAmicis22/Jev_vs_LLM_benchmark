# 02 · Integrazione di Jev con Microsoft Foundry

Questo documento copre due aspetti:

1. come è configurato il **baseline LLM** (GPT-5.4 servito da Microsoft Foundry) per decisioni di
   classificazione affidabili, misurabili e confrontabili con Jev;
2. come **combinare Jev con Foundry** in un'architettura reale: cascata, tool di un agente, quality
   gate, gateway, sicurezza e osservabilità.

## 1. GPT-5.4 su Foundry come classificatore

### Deployment usato nel PoC

| Voce | Valore (dagli header di risposta) |
|---|---|
| Risorsa | `foundry-multimodal.cognitiveservices.azure.com` |
| Deployment | `gpt-5.4` → modello servito `gpt-5.4-2026-03-05` (`x-ms-served-model`) |
| Regione | East US (`x-ms-region`) |
| API | Responses API, `api-version=2025-04-01-preview` |
| Limiti | ~44.590 richieste/min e ~4,46 M token/min (`x-ratelimit-limit-*`) |
| Prezzo | $2,50/M input, $0,25/M input in cache, $15,00/M output ([annuncio Microsoft](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-gpt-5-4-in-microsoft-foundry/4499785)) |

### Structured Outputs: l'LLM può rispondere solo con un'etichetta valida

L'opzione scelta è vincolata da uno **schema JSON strict con `enum`**, costruito dalle stesse opzioni
inviate a Jev. Il decoding è vincolato, quindi niente parsing fragile, nessuna etichetta inventata e
nessun "Sure! The answer is…".

```python
from openai import AsyncAzureOpenAI

client = AsyncAzureOpenAI(azure_endpoint=ENDPOINT, api_key=KEY, api_version="2025-04-01-preview")

response = await client.responses.create(
    model="gpt-5.4",                              # nome del deployment
    input=[
        {"role": "system", "content": system_prompt},   # domanda + elenco opzioni con descrizioni
        {"role": "user", "content": "<customer_message>\n...\n</customer_message>"},
    ],
    text={"format": {
        "type": "json_schema", "name": "intent_routing_decision", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": ["card_arrival", "card_linking", "..."]}},
            "required": ["label"], "additionalProperties": False,
        },
    }},
    reasoning={"effort": "none"},                 # none | low | medium | high | xhigh
    include=["message.output_text.logprobs"],     # solo con effort="none"
    top_logprobs=10,
    store=False,                                  # non conservare la risposta lato servizio
)
label = json.loads(response.output_text)["label"]
```

Per gli scenari ordinali (S3) lo schema è `{"score": {"type": "integer", "enum": [0, 1, 2, 3, 4]}}`.
Implementazione completa: [`systems/foundry.py`](../src/jev_benchmark/systems/foundry.py).

### `reasoning.effort`: la leva latenza/costo/qualità

GPT-5.4 è un modello di ragionamento. Con `effort="none"` risponde subito (adatto a classificazione e
routing); con `low`/`medium`/… genera prima token di ragionamento, fatturati come output a $15/M, che
aumentano latenza e costo. Il benchmark misura due configurazioni: `none` (baseline rapido) e `medium`
(baseline di qualità).

### Una "confidenza" per l'LLM: logprobs

Con `effort="none"` la Responses API restituisce i logprobs dei token generati. La probabilità
dell'etichetta scelta è `exp(Σ logprob)` sui token che compongono il valore nel JSON. I token di
contorno (`{"label":"`) sono forzati dallo schema e valgono ≈ 0. Per gli scenari ordinali si legge
anche la distribuzione sulle cifre 0-4 dai `top_logprobs`. Con il reasoning attivo i logprobs non sono
disponibili, quindi le configurazioni `low`/`medium` non hanno un segnale di confidenza nativo.

### Prompt caching

Azure mette in cache automaticamente i prefissi identici di almeno 1.024 token: l'input in cache
costa $0,25/M invece di $2,50/M. Conviene quindi mettere **prima** la parte costante (istruzioni +
elenco delle opzioni nel system prompt) e **dopo** l'input variabile. Nello scenario S1 (77 opzioni,
~1.800 token) circa il 90% dell'input viene servito dalla cache.

## 2. Architetture di integrazione Jev + Foundry

### A. Cascata: Jev decide, l'LLM interviene solo sui casi incerti

```
richiesta ──► Jev ──► p(scelta) ≥ τ ? ──sì──► decisione di Jev        (ms, frazioni di centesimo)
                              │
                              no
                              ▼
                     GPT-5.4 su Foundry ──► decisione dell'LLM          (secondi, ~$1-10 per 1.000)
```

```python
async def decide(task, state, tau: float):
    jev = await typesafe.system_one(state=state, questions={"d": build_question(task)})
    answer = jev.answers["d"]
    signal = answer.probabilities[answer.choice] if task.kind == "choice" else answer.confidence
    if signal >= tau:
        return answer.choice, "jev"
    llm = await foundry.responses.create(**request_kwargs(task, state))   # stesse opzioni e istruzioni
    return json.loads(llm.output_text)["label"], "llm"
```

- La soglia τ si tara su un set etichettato: il benchmark riporta la curva completa e un punto
  operativo scelto in cross-fitting.
- Variante a **latenza minima**: si chiamano Jev e LLM in parallelo e si usa Jev se è sicuro. La
  latenza diventa quella del più lento solo sui casi incerti, ma si paga sempre l'LLM.
- Variante **a tre livelli**: Jev → GPT-5.4 `none` → GPT-5.4 `medium` o un operatore umano.

### B. Jev come tool di un agente Foundry

Il **Foundry Agent Service** supporta tool definiti da una specifica OpenAPI. Jev può diventare un tool
di decisione rapida dell'agente, per esempio "classifica la richiesta" o "valuta la bozza prima di
inviarla":

- si scrive una spec OpenAPI 3 minimale con l'operazione `POST /v1/systemone` e uno schema di sicurezza
  `apiKey` in header;
- il segreto **non** va nella spec ma in una *project connection* di tipo "Custom keys". Per un'API
  Bearer come Jev: chiave `Authorization`, valore `Bearer <TYPESAFE_API_KEY>`. Il nome dell'header
  nella spec deve coincidere con il nome della chiave nella connessione;
- conviene esporre operazioni **già parametrizzate** (opzioni e istruzioni fisse lato tool, l'agente
  passa solo lo stato), così l'LLM non deve costruire il payload delle domande.

Riferimento: [OpenAPI tool nel Foundry Agent Service](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/openapi).
In alternativa, la stessa logica si espone come *function tool* eseguito dall'applicazione o come
Azure Function, che permette anche di aggiungere cache, logging e soglie.

> Questa architettura è documentata ma non eseguita nel PoC: richiede un progetto Foundry con Agent
> Service e una connessione configurata, mentre il PoC usa direttamente il deployment Azure OpenAI.

### C. Jev come quality gate dell'output generato (scenario S3)

```
Foundry genera la risposta ──► Jev Score "helpfulness" (+ eventuali Noul: tono, policy, PII…)
        ▲                                │
        └── rigenera / escalation ◄── score basso o confidenza bassa
```

È lo scenario dell'*LLM-as-a-judge*: il giudice costa pochi millesimi di dollaro per 1.000 valutazioni
invece di qualche dollaro, e quindi può girare su **tutto** il traffico invece che su un campione. Le
domande atomiche (correttezza, completezza, tono…) si combinano nel codice con il pattern *composite
scoring*.

### D. Gateway, sicurezza e governance

| Tema | Raccomandazione |
|---|---|
| Segreti | chiavi in **Azure Key Vault**; per Azure OpenAI preferire **Entra ID** (keyless) con `DefaultAzureCredential` al posto della `api-key` usata nel PoC |
| Gateway | **Azure API Management** (o LiteLLM, che ha un pass-through TypeSafe) davanti a entrambi i provider: chiavi virtuali, quote, retry, logging centralizzato, cost tracking per team |
| Residenza dei dati | Jev è un **SaaS esterno** (`api.typesafe.ai`, dietro Cloudflare), quindi lo stato inviato esce dal tenant Azure. Prima di usarlo con dati personali o riservati vanno verificati DPA, regione di elaborazione e conservazione dei dati sui [documenti legali TypeSafe](https://docs.typesafe.ai/legal.md), e valutata la minimizzazione o pseudonimizzazione dello stato |
| Disponibilità | Jev è in *early access*: prevedere un fallback automatico sull'LLM (la cascata lo è già) e timeout brevi |
| Versioning | pinnare `jev-1.14.0` e la versione del deployment Foundry. Le soglie della cascata vanno ritarate a ogni cambio di modello |
| Osservabilità | registrare per ogni decisione sistema, versione, probabilità, latenza e costo (es. Application Insights). Il **tasso di escalation** è un indicatore precoce di drift dei dati |

## Riferimenti

- [Introducing GPT-5.4 in Microsoft Foundry](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-gpt-5-4-in-microsoft-foundry/4499785)
- [Reasoning models in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning)
- [OpenAPI tools in Foundry Agent Service](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/openapi)
- Codice: [`systems/foundry.py`](../src/jev_benchmark/systems/foundry.py) ·
  [`systems/jev.py`](../src/jev_benchmark/systems/jev.py) · [`metrics.py`](../src/jev_benchmark/metrics.py) (cascata)
