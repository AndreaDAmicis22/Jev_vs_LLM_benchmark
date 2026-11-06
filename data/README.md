# Dati del benchmark

I file in `benchmark/` sono campioni **stratificati e bilanciati** estratti da dataset pubblici con
`jevbench prepare` (seed 42). Sono versionati per garantire la riproducibilità: chiunque esegua il
benchmark usa esattamente gli stessi esempi. I download originali finiscono in `raw/`, che è escluso
da git e si ricrea con lo stesso comando.

| File | Scenario | Sorgente | Campione | Licenza |
|---|---|---|---|---|
| `intent_routing.jsonl` | S1: classificazione delle richieste in ingresso | [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI), split *test* | 308 = 4 × 77 intent | CC BY 4.0 |
| `workflow_next_step.jsonl` | S2: prossimo passo in un workflow | [ABCD v1.1](https://github.com/asappresearch/abcd) (ASAPP), split *test* | 300 = 10 × 30 azioni, al massimo un punto di decisione per conversazione | MIT |
| `output_quality.jsonl` | S3: qualità di un output generato | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) (NVIDIA), split *validation* | 300 = 60 × 5 livelli di *helpfulness* | CC BY 4.0 |

## Formato

Ogni riga è un `Example`:

```json
{"id": "banking77-test-0042", "label": "card_arrival", "state": {"customer_message": "..."}, "meta": {...}}
```

- `state`: i campi di testo che **entrambi** i sistemi ricevono. Jev li riceve come oggetto JSON,
  l'LLM come sezioni `<campo>...</campo>` nel messaggio utente.
- `label`: la risposta corretta (gold). Per S3 è il livello umano `"0"`…`"4"`.
- `meta`: informazioni di tracciabilità (split, riga o conversazione originale, attributi extra).

## Citazioni

- Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI 2020 (Banking77).
- Chen et al., *Action-Based Conversations Dataset: A Corpus for Building More In-Depth Task-Oriented
  Dialogue Systems*, NAACL 2021 (ABCD).
- Wang et al., *HelpSteer2: Open-source dataset for training top-performing reward models*, 2024.
