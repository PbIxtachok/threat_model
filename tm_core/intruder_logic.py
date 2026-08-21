# -*- coding: utf-8 -*-
"""Правила по нарушителям: уровни возможностей, сговор, актуальность (разделы 5, прил. 3–4)."""
from __future__ import annotations

from typing import Any, Dict, List

from . import dictionaries as dicts

# Возможности сговора (п. 5.9 Методики): вид → виды, с которыми возможен сговор
COLLUSION = {
    1: [5, 6, 7, 8, 9, 10, 11, 12],
    2: [10, 11, 12],
    3: [10, 11, 12],
}


def selected_intruders(intruder_ids: List[int]) -> List[Dict[str, Any]]:
    return [dicts.intruder_by_id(i) for i in sorted(set(intruder_ids))]


def max_level(intruder_ids: List[int]) -> str:
    """Максимальный уровень возможностей среди выбранных нарушителей (Н1..Н4)."""
    levels = [dicts.intruder_by_id(i)["level"] for i in intruder_ids]
    if not levels:
        return "Н1"
    return max(levels, key=lambda s: int(s[1:]))


def collusion_pairs(intruder_ids: List[int]) -> List[Dict[str, Any]]:
    """Пары «вид A может вступать в сговор с видом B» среди выбранных видов."""
    chosen = set(intruder_ids)
    out = []
    for a, partners in COLLUSION.items():
        if a not in chosen:
            continue
        for b in partners:
            if b in chosen:
                out.append({
                    "a": dicts.intruder_by_id(a),
                    "b": dicts.intruder_by_id(b),
                })
    return out


def intruders_by_damage(intruder_ids: List[int], damage_types: List[str]) -> List[Dict[str, Any]]:
    """Актуальные нарушители: у которых цели соотносятся с выбранными видами ущерба."""
    out = []
    for t in selected_intruders(intruder_ids):
        overlap = sorted(set(t.get("damage_types", [])) & set(damage_types))
        out.append({**t, "relevant_damages": overlap, "actual": bool(overlap)})
    return out


def level_info(level: str) -> Dict[str, str]:
    lv = dicts.intruders()["levels"].get(level, {})
    return {"id": level, "title": lv.get("title", ""), "description": lv.get("description", "")}
