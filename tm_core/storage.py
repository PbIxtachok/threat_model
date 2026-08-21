# -*- coding: utf-8 -*-
"""Хранилище: профили, настройки LLM, задания (jobs) с чекпоинтами, кэш LLM."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import JobState, Profile

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PROFILES_DIR = OUTPUT_DIR / "profiles"
JOBS_DIR = OUTPUT_DIR / "jobs"
SETTINGS_FILE = OUTPUT_DIR / "llm_settings.json"
CACHE_FILE = OUTPUT_DIR / "llm_cache.json"
DRAFT_FILE = OUTPUT_DIR / "draft_profile.json"
SAMPLE_PROFILES_DIR = DATA_DIR / "sample_profiles"

_lock = threading.Lock()


def seed_sample_profiles() -> None:
    """Копирует готовые примеры профилей из data/sample_profiles/ в
    PROFILES_DIR — только если файл с таким именем там ещё не существует
    (пользовательские правки не затираются). Отсутствие каталога с
    примерами — не ошибка."""
    if not SAMPLE_PROFILES_DIR.is_dir():
        return
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(SAMPLE_PROFILES_DIR.glob("*.json")):
        dst = PROFILES_DIR / src.name
        if not dst.exists():
            try:
                shutil.copy2(src, dst)
            except Exception:
                logger.exception("Не удалось скопировать пример профиля %s", src.name)


def ensure_dirs() -> None:
    for p in (OUTPUT_DIR, PROFILES_DIR, JOBS_DIR):
        p.mkdir(parents=True, exist_ok=True)
    seed_sample_profiles()


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\wа-яА-ЯёЁ\- ]", "_", name).strip().replace(" ", "_")
    return name[:80] or "profile"


# ----------------------------- профили --------------------------------
def save_profile(profile: Profile, name: Optional[str] = None) -> Path:
    ensure_dirs()
    now = datetime.now().isoformat(timespec="seconds")
    if not profile.created_at:
        profile.created_at = now
    profile.updated_at = now
    fname = _safe_name(name or profile.object_name)
    path = PROFILES_DIR / f"{fname}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def load_profile(path: str | Path) -> Profile:
    with open(path, "r", encoding="utf-8") as f:
        return Profile.from_dict(json.load(f))


def list_profiles() -> List[str]:
    ensure_dirs()
    return [str(p) for p in sorted(PROFILES_DIR.glob("*.json"))]


# --------------------------- настройки LLM ----------------------------
def save_llm_settings(cfg_dict: Dict[str, Any]) -> None:
    """Сохраняет настройки LLM.

    Ожидаемый формат: ``{"analysis": {LLMConfig...}, "text": {LLMConfig...}}``.
    """
    ensure_dirs()
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
    tmp.replace(SETTINGS_FILE)


def load_llm_settings() -> Dict[str, Any]:
    """Читает настройки LLM в формате ``{"analysis": {...}, "text": {...}}``.

    Обратная совместимость: плоский dict со старым форматом (ключ "provider")
    трактуется как единая конфигурация для обеих ролей.
    """
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.exception("Не удалось прочитать настройки LLM")
            return {}
        if isinstance(data, dict) and "provider" in data:
            return {"analysis": data, "text": data}
        if isinstance(data, dict):
            out: Dict[str, Any] = {}
            if isinstance(data.get("analysis"), dict):
                out["analysis"] = data["analysis"]
            if isinstance(data.get("text"), dict):
                out["text"] = data["text"]
            if "text" not in out and "analysis" in out:
                out["text"] = out["analysis"]
            return out
    return {}


# ------------------------- черновик профиля ----------------------------
def save_draft(data: Dict[str, Any]) -> None:
    """Автосохранение черновика профиля (атомарная запись через tmp)."""
    ensure_dirs()
    tmp = DRAFT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(DRAFT_FILE)


def load_draft() -> Optional[Dict[str, Any]]:
    """Возвращает сохранённый черновик профиля либо None."""
    if DRAFT_FILE.exists():
        try:
            with open(DRAFT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Не удалось прочитать черновик профиля")
    return None


# ------------------------------ кэш LLM -------------------------------
class LLMCache:
    """Дисковый кэш ответов LLM со стабильными sha256-ключами."""

    def __init__(self, path: Path = CACHE_FILE):
        self.path = path
        self._data: Dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                logger.exception("Кэш LLM повреждён — начинаем с пустого")
                self._data = {}
        self._loaded = True

    @staticmethod
    def key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(str(p).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> Optional[str]:
        with _lock:
            self._load()
            return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with _lock:
            self._load()
            self._data[key] = value
            ensure_dirs()
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            tmp.replace(self.path)


# ------------------------------ задания -------------------------------
def new_job(profile: Profile, profile_name: str) -> JobState:
    ensure_dirs()
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    state = JobState.new(job_id, profile, profile_name)
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    save_job_state(state)
    return state


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def save_job_state(state: JobState) -> None:
    state.updated_at = datetime.now().isoformat(timespec="seconds")
    d = job_dir(state.job_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
    tmp.replace(d / "state.json")


def load_job_state(job_id: str) -> Optional[JobState]:
    p = job_dir(job_id) / "state.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return JobState.from_dict(json.load(f))


def load_job_profile(job_id: str) -> Optional[Profile]:
    p = job_dir(job_id) / "profile.json"
    if not p.exists():
        return None
    return load_profile(p)


def list_jobs() -> List[JobState]:
    ensure_dirs()
    out: List[JobState] = []
    for d in sorted(JOBS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "state.json").exists():
            st = load_job_state(d.name)
            if st:
                out.append(st)
    return out


# --------------------- чекпоинты результатов этапов -------------------
def save_stage_result(job_id: str, stage: str, result: Dict[str, Any]) -> None:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{stage}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    tmp.replace(d / f"{stage}.json")


def load_stage_result(job_id: str, stage: str) -> Optional[Dict[str, Any]]:
    p = job_dir(job_id) / f"{stage}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_stage_result(job_id: str, stage: str) -> None:
    """Удаляет файл результата этапа (``<stage>.json``), если он существует.

    Partial-чекпоинты батчей (``<stage>.partial.jsonl``) НЕ удаляются —
    накопленный прогресс этапа сохраняется.
    """
    p = job_dir(job_id) / f"{stage}.json"
    p.unlink(missing_ok=True)


# --- построчные partial-чекпоинты внутри «тяжёлых» этапов (jsonl) ------
def partial_path(job_id: str, stage: str) -> Path:
    return job_dir(job_id) / f"{stage}.partial.jsonl"


def append_partial(job_id: str, stage: str, item: Dict[str, Any]) -> None:
    with open(partial_path(job_id, stage), "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_partials(job_id: str, stage: str) -> List[Dict[str, Any]]:
    p = partial_path(job_id, stage)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Пропущена повреждённая строка partial: %s", line[:100])
    return out
