# -*- coding: utf-8 -*-
"""Лёгкий RAG по тексту Методики оценки угроз ФСТЭК России (data/methodology.txt).

Гибрид «промпт-пак + BM25-RAG»: текст Методики режется на фрагменты по
заголовкам (строки с префиксом "## "), мелкие фрагменты склеиваются до
~1500–2500 символов, а релевантные выдержки подбираются готовым BM25 из
``tm_core/retrieval.py`` — без новых тяжёлых зависимостей.

Если файл отсутствует, ``load_chunks()`` возвращает [] и ``excerpts_for()``
возвращает "" — промпты работают как раньше, без выдержек (info-лог).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .retrieval import BM25

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
METHODOLOGY_FILE = BASE_DIR / "data" / "methodology.txt"

# Целевой размер фрагмента: мелкие склеиваем, пока не превысим MAX_CHUNK.
MIN_CHUNK = 1500
MAX_CHUNK = 2500

_chunks_cache: Dict[str, List[str]] = {}
_index_cache: Dict[str, BM25] = {}


def _split_chunks(text: str) -> List[str]:
    """Режет текст по заголовкам "## " и склеивает мелкие фрагменты."""
    parts: List[str] = []
    cur: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur:
                parts.append("\n".join(cur).strip())
            cur = [line]
        else:
            cur.append(line)
    if cur:
        parts.append("\n".join(cur).strip())

    merged: List[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        cand = (buf + "\n\n" + part) if buf else part
        if buf and len(cand) > MAX_CHUNK:
            merged.append(buf)
            buf = part
        else:
            buf = cand
    if buf:
        merged.append(buf)
    return merged


def load_chunks() -> List[str]:
    """Читает data/methodology.txt и возвращает список фрагментов.

    Файл отсутствует → [] (fallback: промпты без выдержек, info-лог).
    Результат кэшируется по пути к файлу.
    """
    path = METHODOLOGY_FILE
    key = str(path)
    if key in _chunks_cache:
        return _chunks_cache[key]
    if not path.exists():
        logger.info("Файл Методики %s не найден — промпты без выдержек (RAG отключён).", path)
        _chunks_cache[key] = []
        return []
    chunks = _split_chunks(path.read_text(encoding="utf-8"))
    logger.info("Методика загружена: %d фрагментов из %s", len(chunks), path.name)
    _chunks_cache[key] = chunks
    return chunks


def _index() -> Optional[BM25]:
    chunks = load_chunks()
    if not chunks:
        return None
    key = str(METHODOLOGY_FILE)
    if key not in _index_cache:
        _index_cache[key] = BM25(chunks)
    return _index_cache[key]


def excerpts_for(query: str, k: int = 4, max_chars: int = 4500) -> str:
    """Подбирает k наиболее релевантных фрагментов Методики по запросу.

    Склейка через разделитель, обрезка по max_chars; пустой индекс → "".
    """
    idx = _index()
    if idx is None:
        return ""
    chunks = _chunks_cache[str(METHODOLOGY_FILE)]
    out: List[str] = []
    total = 0
    for i, score in idx.top_n(query, n=k):
        if score <= 0:
            continue
        piece = chunks[i].strip()
        if out and total + len(piece) > max_chars:
            break
        out.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(out)[:max_chars]
