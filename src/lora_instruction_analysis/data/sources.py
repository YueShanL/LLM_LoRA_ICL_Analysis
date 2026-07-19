"""Public text sources used to seed synthetic transformations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator


@dataclass(frozen=True)
class PublicTextSource:
    source_id: str
    hf_path: str
    hf_name: str | None
    split: str
    text_field: str
    citation: str


PUBLIC_TEXT_SOURCES: dict[str, PublicTextSource] = {
    "wikitext": PublicTextSource(
        source_id="wikitext",
        hf_path="Salesforce/wikitext",
        hf_name="wikitext-2-raw-v1",
        split="train",
        text_field="text",
        citation="Hugging Face datasets: Salesforce/wikitext, wikitext-2-raw-v1",
    ),
    "rotten_tomatoes": PublicTextSource(
        source_id="rotten_tomatoes",
        hf_path="cornell-movie-review-data/rotten_tomatoes",
        hf_name=None,
        split="train",
        text_field="text",
        citation="Hugging Face datasets: cornell-movie-review-data/rotten_tomatoes",
    ),
    "ag_news": PublicTextSource(
        source_id="ag_news",
        hf_path="fancyzhx/ag_news",
        hf_name=None,
        split="train",
        text_field="text",
        citation="Hugging Face datasets: fancyzhx/ag_news",
    ),
}


def list_sources() -> list[PublicTextSource]:
    return list(PUBLIC_TEXT_SOURCES.values())


def get_source(source_id: str) -> PublicTextSource:
    try:
        return PUBLIC_TEXT_SOURCES[source_id]
    except KeyError as exc:
        available = ", ".join(sorted(PUBLIC_TEXT_SOURCES))
        raise KeyError(f"Unknown source_id {source_id!r}. Available sources: {available}") from exc


def normalize_text(text: str, min_words: int, max_words: int) -> str | None:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    words = text.split()
    if len(words) < min_words:
        return None
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if not re.search(r"[A-Za-z]", text):
        return None
    return text


def iter_public_texts(
    source: PublicTextSource,
    *,
    max_source_rows: int,
    min_words: int,
    max_words: int,
    streaming: bool = False,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to download public text sources. "
            "Install with `venv\\Scripts\\python.exe -m pip install -e .`."
        ) from exc

    kwargs = {"split": source.split, "streaming": streaming}
    dataset = load_dataset(source.hf_path, source.hf_name, **kwargs)
    seen = 0
    for row in dataset:
        if seen >= max_source_rows:
            break
        seen += 1
        value = row.get(source.text_field)
        if not isinstance(value, str):
            continue
        normalized = normalize_text(value, min_words=min_words, max_words=max_words)
        if normalized is not None:
            yield normalized
