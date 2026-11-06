"""Shared abstractions for the three benchmark scenarios.

A `Task` holds everything that defines the decision: the question (`instructions`) and the ordered
set of options with their descriptions (`labels`). Both systems receive exactly the same text:

* Jev gets `instructions` + `labels` as a `Choice`/`Score` question and the example `state` as JSON.
* The LLM gets `instructions` + `labels` in the system prompt and the same `state` rendered as
  XML-tagged sections in the user message; its answer is constrained by a JSON schema enum.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, Field

from jev_benchmark.config import BENCHMARK_DIR

T = TypeVar("T")


class TaskKind(StrEnum):
    """How the decision is modelled, and therefore which Jev primitive is used."""

    CHOICE = "choice"  # one option out of N            -> Jev `Choice`
    ORDINAL = "ordinal"  # ordered levels "0".."K-1"     -> Jev `Score`


class Example(BaseModel):
    """One benchmark item. `state` holds named text fields shown to both systems."""

    id: str
    label: str
    state: dict[str, str]
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    kind: TaskKind
    instructions: str
    labels: dict[str, str]
    dataset: str

    @property
    def label_names(self) -> list[str]:
        return list(self.labels)

    @property
    def path(self) -> Path:
        return BENCHMARK_DIR / f"{self.name}.jsonl"

    def load(self, limit: int | None = None) -> list[Example]:
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found: run `jevbench prepare` first")
        with self.path.open(encoding="utf-8") as fh:
            examples = [Example.model_validate_json(line) for line in fh if line.strip()]
        return examples[:limit] if limit else examples

    def save(self, examples: Iterable[Example]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as fh:
            for ex in examples:
                if ex.label not in self.labels:
                    raise ValueError(f"{ex.id}: label {ex.label!r} is not an option of {self.name}")
                fh.write(ex.model_dump_json() + "\n")
        return self.path


def render_state_text(state: dict[str, str]) -> str:
    """Render the state for the LLM: same fields Jev receives as JSON, as XML-tagged sections."""
    return "\n\n".join(f"<{key}>\n{value}\n</{key}>" for key, value in state.items())


def stratified_sample(items: Sequence[T], key: Callable[[T], str], per_class: int, seed: int) -> list[T]:
    """Pick `per_class` items for every class (all of them if fewer), deterministically."""
    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    rng = random.Random(seed)
    picked: list[T] = []
    for cls in sorted(groups):
        bucket = list(groups[cls])
        rng.shuffle(bucket)
        picked.extend(bucket[:per_class])
    rng.shuffle(picked)
    return picked


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest` once (no-op if the file already exists)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    tmp.replace(dest)
    return dest
