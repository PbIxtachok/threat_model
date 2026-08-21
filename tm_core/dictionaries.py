# -*- coding: utf-8 -*-
"""Загрузка справочников (данные извлечены из Методики ФСТЭК от 05.02.2021)."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DICT_DIR = DATA_DIR / "dictionaries"


@lru_cache(maxsize=None)
def _load_json(name: str) -> Dict[str, Any]:
    with open(DICT_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def intruders() -> Dict[str, Any]:
    """Виды нарушителей (табл. 6.1) и уровни возможностей (табл. 8.1)."""
    return _load_json("intruders.json")


def damages() -> Dict[str, Any]:
    """Виды риска (ущерба) и типовые негативные последствия (табл. 4.1)."""
    return _load_json("damages.json")


def misc() -> Dict[str, Any]:
    """Прочие справочники (типы систем, компоненты, интерфейсы и т.д.)."""
    return _load_json("misc.json")


def security_tools() -> Dict[str, Any]:
    """Справочник типовых СЗИ/СКЗИ (категории → items: name/vendor/description/crypto)."""
    return _load_json("security_tools.json")


def security_tools_choices() -> List[str]:
    """Плоский список для UI: каждая запись — «{name} ({vendor})».

    Порядок соответствует порядку категорий и записей в справочнике,
    принадлежность к категории сохраняется (восстанавливается поиском).
    """
    out: List[str] = []
    for cat in security_tools().get("categories", []):
        for item in cat.get("items", []):
            name = str(item.get("name", "")).strip()
            vendor = str(item.get("vendor", "")).strip()
            if not name:
                continue
            out.append(f"{name} ({vendor})" if vendor else name)
    return out


def is_crypto_tool(choice_str: str) -> bool:
    """True, если строка выбора UI соответствует средству с crypto=true."""
    s = str(choice_str or "").strip()
    if not s:
        return False
    for cat in security_tools().get("categories", []):
        for item in cat.get("items", []):
            name = str(item.get("name", "")).strip()
            vendor = str(item.get("vendor", "")).strip()
            label = f"{name} ({vendor})" if vendor else name
            if s == label or s == name:
                return bool(item.get("crypto"))
    return False


def intruder_by_id(iid: int) -> Dict[str, Any]:
    for t in intruders()["types"]:
        if t["id"] == iid:
            return t
    raise KeyError(f"Неизвестный вид нарушителя: {iid}")


# ------------------------------ датасеты ------------------------------
@lru_cache(maxsize=None)
def load_ubi() -> List[Dict[str, str]]:
    """УБИ из БДУ ФСТЭК: Number|text|description."""
    out: List[Dict[str, str]] = []
    with open(DATA_DIR / "ubi.csv", "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="|"):
            n = str(row.get("Number", "")).strip()
            if not n:
                continue
            out.append({
                "Number": n,
                "text": str(row.get("text", "")).strip(),
                "description": str(row.get("description", "")).strip(),
            })
    return out


@lru_cache(maxsize=None)
def load_tactics() -> List[Dict[str, str]]:
    """Тактики/способы реализации: Module|Category|number|Description."""
    out: List[Dict[str, str]] = []
    with open(DATA_DIR / "threat_list.csv", "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="|"):
            num = str(row.get("number", "")).strip()
            if not num:
                continue
            out.append({
                "Module": str(row.get("Module", "")).strip(),
                "Category": str(row.get("Category", "")).strip(),
                "number": num,
                "Description": str(row.get("Description", "")).strip(),
            })
    return out
