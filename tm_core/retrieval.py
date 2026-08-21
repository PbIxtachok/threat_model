# -*- coding: utf-8 -*-
"""Лёгкий BM25 без внешних зависимостей (заменяет langchain + FAISS + HF-эмбеддинги).

Для задачи shortlist-а по короткому запросу качества BM25 достаточно,
а сервис перестаёт тянуть ~5 ГБ зависимостей и грузить эмбеддинг-модель.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[а-яa-zё0-9]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf: List[Counter] = [Counter(t) for t in self.doc_tokens]
        self.df: Counter = Counter()
        for c in self.tf:
            for term in c:
                self.df[term] += 1
        self.n = len(self.doc_tokens)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out: List[float] = []
        for i in range(self.n):
            s = 0.0
            dl = self.doc_len[i] or 1
            for term in q:
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                idf = self._idf(term)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top_n(self, query: str, n: int = 10) -> List[Tuple[int, float]]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
        return [(i, sc[i]) for i in order[:n]]
