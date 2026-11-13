"""S4 - Incoming request classification in Italian (MASSIVE it-IT).

MASSIVE (Amazon, FitzGerald et al. 2023) contains requests to a virtual assistant localized into 51
languages by professional native speakers (not machine translation). We use the it-IT locale: the
system must assign each Italian request to one of 60 intents. Unlike S1, instructions and option
descriptions are written in Italian too, as an Italian deployment would have them; the descriptions
were written from the *train* split only.
"""

from __future__ import annotations

import json
import random
import tarfile
from collections import Counter, defaultdict

from jev_benchmark.config import RAW_DIR
from jev_benchmark.tasks.base import Example, Task, TaskKind, balanced_quotas, download

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
LOCALE_MEMBER = "1.1/data/it-IT.jsonl"

# Alphabetical order of the 60 MASSIVE intents (canonical order of the options).
INTENTS: dict[str, str] = {
    "alarm_query": "Chiede quali sveglie sono impostate o informazioni su una sveglia esistente.",
    "alarm_remove": "Vuole cancellare o disattivare una o più sveglie.",
    "alarm_set": "Vuole impostare una nuova sveglia o un avviso a un certo orario.",
    "audio_volume_down": "Vuole abbassare il volume.",
    "audio_volume_mute": "Vuole silenziare l'audio o far tacere l'assistente.",
    "audio_volume_other": "Altre richieste sul volume o sull'audio (impostazioni, regolazioni, chiarezza).",
    "audio_volume_up": "Vuole alzare il volume.",
    "calendar_query": "Chiede quali eventi, appuntamenti o promemoria ci sono in calendario.",
    "calendar_remove": "Vuole cancellare un evento, una riunione o un promemoria dal calendario.",
    "calendar_set": "Vuole aggiungere un evento, un appuntamento o un promemoria al calendario.",
    "cooking_query": "Domanda generica di cucina (cosa cucinare, sostituire un ingrediente, come cuocere).",
    "cooking_recipe": "Chiede una ricetta o le istruzioni per preparare un piatto.",
    "datetime_convert": "Vuole convertire un orario tra fusi orari diversi.",
    "datetime_query": "Chiede l'ora o la data attuale, anche in un'altra città o fuso orario.",
    "email_addcontact": "Vuole aggiungere o modificare un contatto o un indirizzo email in rubrica.",
    "email_query": "Vuole controllare la posta o cercare email ricevute.",
    "email_querycontact": "Chiede i dati di un contatto in rubrica (numero, indirizzo, email).",
    "email_sendemail": "Vuole scrivere, inviare o rispondere a una email.",
    "general_greet": "Saluta o fa una domanda di cortesia all'assistente (ciao, come stai, chi sei).",
    "general_joke": "Chiede una barzelletta o qualcosa per ridere.",
    "general_quirky": "Chiacchiere, frasi strane o richieste che non rientrano negli altri ambiti.",
    "iot_cleaning": "Vuole avviare o comandare il robot aspirapolvere o la pulizia della casa.",
    "iot_coffee": "Vuole che la macchina del caffè prepari una bevanda.",
    "iot_hue_lightchange": "Vuole cambiare colore o tonalità delle luci.",
    "iot_hue_lightdim": "Vuole abbassare l'intensità delle luci.",
    "iot_hue_lightoff": "Vuole spegnere le luci.",
    "iot_hue_lighton": "Vuole accendere le luci.",
    "iot_hue_lightup": "Vuole aumentare la luminosità delle luci.",
    "iot_wemo_off": "Vuole spegnere una presa smart.",
    "iot_wemo_on": "Vuole accendere una presa smart.",
    "lists_createoradd": "Vuole creare una lista o aggiungere un elemento a una lista.",
    "lists_query": "Chiede quali liste esistono o cosa contiene una lista.",
    "lists_remove": "Vuole eliminare una lista o togliere un elemento da una lista.",
    "music_dislikeness": "Dice che una canzone o un genere non gli piace, o di non riprodurlo più.",
    "music_likeness": "Dice che una canzone o un genere gli piace, o vuole salvarlo tra i preferiti.",
    "music_query": "Chiede informazioni sulla musica (titolo della canzone, artista, album).",
    "music_settings": "Vuole cambiare le impostazioni di riproduzione (ripeti, casuale, cambia canzone).",
    "news_query": "Chiede notizie o titoli di attualità.",
    "play_audiobook": "Vuole ascoltare o riprendere un audiolibro.",
    "play_game": "Vuole giocare a un gioco con l'assistente.",
    "play_music": "Vuole ascoltare musica (una canzone, un artista, un genere, una playlist).",
    "play_podcasts": "Vuole ascoltare un podcast o un suo episodio.",
    "play_radio": "Vuole ascoltare una stazione radio.",
    "qa_currency": "Chiede un tasso di cambio o una conversione tra valute.",
    "qa_definition": "Chiede il significato o la definizione di una parola o di un concetto.",
    "qa_factoid": "Domanda di cultura generale su fatti, persone o luoghi.",
    "qa_maths": "Chiede di fare un calcolo matematico.",
    "qa_stock": "Chiede quotazioni o andamento di azioni e borsa.",
    "recommendation_events": "Chiede consigli su eventi o iniziative in zona.",
    "recommendation_locations": "Chiede consigli su luoghi (ristoranti, negozi, alloggi) in una zona.",
    "recommendation_movies": "Chiede consigli su film da vedere o in programmazione.",
    "social_post": "Vuole pubblicare un post o inviare un messaggio o reclamo su un social network.",
    "social_query": "Chiede aggiornamenti o novità dai social network.",
    "takeaway_order": "Vuole ordinare cibo da asporto o a domicilio.",
    "takeaway_query": "Chiede se un locale fa asporto o consegna a domicilio, o informazioni su un ordine.",
    "transport_query": "Chiede orari, percorsi o indicazioni per spostarsi o raggiungere un luogo.",
    "transport_taxi": "Vuole chiamare o prenotare un taxi o un passaggio.",
    "transport_ticket": "Vuole cercare o comprare un biglietto di treno, aereo o altro mezzo.",
    "transport_traffic": "Chiede informazioni sul traffico o sui tempi di percorrenza in auto.",
    "weather_query": "Chiede le previsioni del tempo o informazioni sul meteo.",
}

TASK = Task(
    name="intent_routing_it",
    title="S4 - Classificazione delle richieste in italiano",
    kind=TaskKind.CHOICE,
    instructions=(
        "Un utente ha rivolto questa richiesta a un assistente virtuale. "
        "Quale intent descrive meglio ciò che l'utente vuole?"
    ),
    labels=INTENTS,
    dataset=(
        "MASSIVE 1.1 it-IT (Amazon, FitzGerald et al. 2023), split test (+dev per gli intent rari), "
        "CC BY 4.0"
    ),
)


def _load_locale() -> list[dict]:
    """Rows of the it-IT locale (extracted once from the official archive)."""
    cached = RAW_DIR / "massive" / "it-IT.jsonl"
    if not cached.exists():
        archive = download(MASSIVE_URL, RAW_DIR / "massive" / "amazon-massive-dataset-1.1.tar.gz")
        with tarfile.open(archive) as tf:
            member = tf.extractfile(LOCALE_MEMBER)
            if member is None:
                raise FileNotFoundError(f"{LOCALE_MEMBER} not found in {archive}")
            cached.write_bytes(member.read())
    with cached.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(total: int = 500, seed: int = 42) -> list[Example]:
    """`total` requests as balanced across intents as the evaluation splits allow (test, then dev)."""
    rows = [r for r in _load_locale() if r["partition"] in ("test", "dev")]
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"test": [], "dev": []})
    for row in rows:
        pools[row["intent"]][row["partition"]].append(row)
    unknown = set(pools) - set(INTENTS)
    if unknown:
        raise ValueError(f"Intents missing from the option list: {sorted(unknown)}")

    quotas = balanced_quotas(Counter(r["intent"] for r in rows), total, seed)
    rng = random.Random(seed)
    picked: list[dict] = []
    for intent in sorted(pools):
        candidates = []
        for split in ("test", "dev"):
            bucket = pools[intent][split]
            rng.shuffle(bucket)
            candidates.extend(bucket)
        picked.extend(candidates[: quotas[intent]])
    rng.shuffle(picked)
    return [
        Example(
            id=f"massive-it-{row['partition']}-{row['id']}",
            label=row["intent"],
            state={"richiesta_utente": row["utt"].strip()},
            meta={"split": row["partition"], "massive_id": row["id"], "scenario": row["scenario"]},
        )
        for row in picked
    ]
