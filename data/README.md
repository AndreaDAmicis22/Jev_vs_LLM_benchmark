# Dati del benchmark

I file in `benchmark/` sono campioni **stratificati** di dataset pubblici, estratti con
`jevbench prepare` (seed 42, **500 esempi per scenario**). Sono versionati per garantire la
riproducibilità: chiunque esegua il benchmark usa esattamente gli stessi esempi. I download originali
finiscono in `raw/`, escluso da git e ricreato dallo stesso comando.

| File | Scenario | Sorgente | Campione | Licenza |
|---|---|---|---|---|
| `intent_routing.jsonl` | S1: classificazione delle richieste in ingresso | [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI), split *test* | 500, 6-7 per ciascuno dei 77 intent | CC BY 4.0 |
| `workflow_next_step.jsonl` | S2: prossimo passo in un workflow | [ABCD v1.1](https://github.com/asappresearch/abcd) (ASAPP), split *test* | 500, 16-17 per ciascuna delle 30 azioni, al massimo un punto di decisione per conversazione | MIT |
| `output_quality.jsonl` | S3: qualità di un output generato | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) (NVIDIA), split *validation* | 500: tutti i 78 esempi di livello 0 e i 93 di livello 1, ~110 per i livelli 2-4 | CC BY 4.0 |
| `intent_routing_it.jsonl` | S4: classificazione delle richieste in italiano | [MASSIVE 1.1](https://github.com/alexa/massive) (Amazon), locale *it-IT*, split *test* (486) + *dev* (14, solo per gli intent rari) | 500, fino a 9 per ciascuno dei 60 intent | CC BY 4.0 |

## Versioni del campione

- **v1** (6 novembre 2026): 308 / 300 / 300 esempi per S1 / S2 / S3.
- **v2** (13 novembre 2026): 500 esempi per scenario, con S4 aggiunto. Ogni campione v2 **contiene
  integralmente** il campione v1, con testo ed etichetta identici, verificato anche da
  `tests/test_tasks.py`. Per questo i risultati già misurati sulla v1 restano validi e `jevbench run`
  esegue solo gli esempi nuovi.

Il campionamento è bilanciato quanto la disponibilità consente: le quote sono distribuite in modo
uniforme e le classi con pochi esempi vengono prese per intero (HelpSteer2 livelli 0-1, alcuni intent
MASSIVE).

## Formato

Ogni riga è un `Example`:

```json
{"id": "banking77-test-0042", "label": "card_arrival", "state": {"customer_message": "..."}, "meta": {...}}
```

- `state`: i campi di testo che **entrambi** i sistemi ricevono. Jev li riceve come oggetto JSON,
  l'LLM come sezioni `<campo>…</campo>` nel messaggio utente.
- `label`: la risposta corretta (gold). Per S3 è il livello umano `"0"`…`"4"`.
- `meta`: informazioni di tracciabilità (split, riga o conversazione originale, attributi extra).

## Citazioni

- Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI 2020 (Banking77).
- Chen et al., *Action-Based Conversations Dataset: A Corpus for Building More In-Depth Task-Oriented
  Dialogue Systems*, NAACL 2021 (ABCD).
- Wang et al., *HelpSteer2: Open-source dataset for training top-performing reward models*, 2024.
- FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51
  Typologically-Diverse Languages*, ACL 2023.
