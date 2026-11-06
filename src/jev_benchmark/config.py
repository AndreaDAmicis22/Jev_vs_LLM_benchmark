"""Settings loaded from `.env`, project paths and pricing assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BENCHMARK_DIR = DATA_DIR / "benchmark"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


class Settings(BaseSettings):
    """Endpoints and secrets, read from environment variables or from `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    azure_openai_endpoint: str = ""
    azure_openai_api_key: SecretStr = SecretStr("")
    azure_openai_api_version: str = "2025-04-01-preview"
    azure_openai_deployment: str = "gpt-5.4"

    typesafe_api_key: SecretStr = SecretStr("")
    jev_model: str = "jev-1.14.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class Pricing:
    """List prices in USD per million tokens."""

    input: float
    cached_input: float
    output: float
    source: str

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """Cost of one request. Reasoning tokens are billed as output tokens."""
        uncached = max(input_tokens - cached_tokens, 0)
        return (
            uncached * self.input + cached_tokens * self.cached_input + output_tokens * self.output
        ) / 1_000_000


JEV_PRICING = Pricing(
    input=0.042,
    cached_input=0.042,
    output=0.0,
    source="https://docs.typesafe.ai/models.md (accessed 2026-11-06): $42 per billion input tokens, "
    "output tokens free",
)

GPT54_PRICING = Pricing(
    input=2.50,
    cached_input=0.25,
    output=15.00,
    source="https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/"
    "introducing-gpt-5-4-in-microsoft-foundry/4499785 (Global Standard): $2.50/M input, "
    "$0.25/M cached input, $15.00/M output",
)
