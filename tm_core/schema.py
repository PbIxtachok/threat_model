# -*- coding: utf-8 -*-
"""Схемы данных: профиль информационной системы и состояние задания генерации."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Component:
    """Компонент ИС — объект воздействия."""
    name: str = ""
    ctype: str = ""            # тип из справочника component_types
    purpose: str = ""          # назначение
    location: str = ""         # размещение (сегмент/площадка)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Component":
        return Component(
            name=str(d.get("name", "")).strip(),
            ctype=str(d.get("ctype", "")).strip(),
            purpose=str(d.get("purpose", "")).strip(),
            location=str(d.get("location", "")).strip(),
        )


@dataclass
class Profile:
    """Профиль объекта (исходные данные для модели угроз)."""
    # --- Реквизиты документа (раздел 1) ---
    object_name: str = ""                 # наименование ИС
    operator_name: str = ""               # оператор / обладатель информации
    responsible: str = ""                 # ответственное подразделение / лицо
    developer_org: str = ""               # организация-разработчик модели угроз (опц.)
    approver_position: str = ""           # должность утверждающего
    city: str = ""
    year: str = ""

    # --- Классификация (раздел 2) ---
    system_type: str = ""                 # ИСПДн / ГИС / ...
    protection_level: str = ""            # УЗ-1..УЗ-4 / К1..К3
    classification_basis: str = ""        # реквизиты акта классификации (опц.)
    purpose: str = ""                     # назначение и задачи ИС
    business_processes: str = ""          # основные (бизнес-)процессы
    scale: str = ""                       # одноплощадочная/многоплощадочная, адреса

    # --- Обрабатываемая информация ---
    info_kinds: List[str] = field(default_factory=list)
    pdn_categories: List[str] = field(default_factory=list)
    pdn_subjects: List[str] = field(default_factory=list)
    pdn_volume: str = ""

    # --- Архитектура ---
    components: List[Component] = field(default_factory=list)
    network_interfaces: List[str] = field(default_factory=list)
    has_internet: bool = False
    has_wireless: bool = False
    has_remote_access: bool = False
    has_contractors: bool = False
    has_external_integrations: bool = False
    cloud_model: str = "Не используется"
    cloud_details: str = ""               # разграничение ответственности с провайдером
    security_tools: List[str] = field(default_factory=list)  # применяемые СЗИ/СКЗИ (список)
    has_crypto: bool = False              # применяются СКЗИ (криптографическая защита)
    architecture_notes: str = ""          # свободное описание архитектуры

    # --- Пользователи ---
    user_groups: List[str] = field(default_factory=list)
    users_notes: str = ""

    # --- Негативные последствия ---
    damage_types: List[str] = field(default_factory=list)      # ["У1", "У2", ...]
    consequences: List[str] = field(default_factory=list)      # выбранные типовые последствия
    consequences_custom: str = ""                               # свои последствия (по строке)

    # --- Нарушители ---
    intruder_ids: List[int] = field(default_factory=list)      # id из intruders.json
    intruders_excluded_reason: str = ""                        # обоснование исключений (опц.)

    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Profile":
        d = dict(d or {})
        comps = [Component.from_dict(c) for c in d.get("components", []) if isinstance(c, dict)]
        known = {f for f in Profile.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in known and k != "components"}
        # обратная совместимость: в старых профилях security_tools — строка
        # со средствами через «;» — разбиваем её в список
        st = clean.get("security_tools")
        if isinstance(st, str):
            clean["security_tools"] = [s.strip() for s in st.split(";") if s.strip()]
        elif st is None:
            clean["security_tools"] = []
        p = Profile(**clean)
        p.components = comps
        return p

    def fingerprint(self) -> str:
        """Стабильный отпечаток профиля (для кэша LLM и привязки заданий)."""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.object_name.strip():
            errors.append("Не указано наименование объекта (ИС).")
        if not self.operator_name.strip():
            errors.append("Не указан оператор (обладатель информации).")
        if not self.system_type:
            errors.append("Не выбран тип системы.")
        if not self.info_kinds:
            errors.append("Не выбран состав обрабатываемой информации.")
        if not self.damage_types:
            errors.append("Не выбраны виды риска (ущерба) У1–У3.")
        if not self.consequences and not self.consequences_custom.strip():
            errors.append("Не выбраны негативные последствия.")
        if not self.intruder_ids:
            errors.append("Не отмечен ни один вид нарушителя.")
        if not self.components:
            errors.append("Не задан ни один компонент ИС (объект воздействия).")
        if self.has_remote_access and not self.network_interfaces:
            errors.append("Для удаленного доступа укажите интерфейсы/каналы.")
        return errors

    def summary(self) -> str:
        """Краткая текстовая сводка профиля для промптов LLM."""
        comp_lines = "; ".join(
            f"{c.name} ({c.ctype})" for c in self.components[:20]
        )
        parts = [
            f"Объект: {self.object_name}",
            f"Тип системы: {self.system_type}; уровень/класс защищённости: {self.protection_level or 'не задан'}",
            f"Назначение: {self.purpose or '—'}",
            f"Обрабатываемая информация: {', '.join(self.info_kinds) or '—'}",
            f"Категории ПДн: {', '.join(self.pdn_categories) or '—'}",
            f"Компоненты: {comp_lines or '—'}",
            f"Интерфейсы: {', '.join(self.network_interfaces) or '—'}",
            f"Подключение к Интернет: {'да' if self.has_internet else 'нет'}",
            f"Беспроводные сети: {'да' if self.has_wireless else 'нет'}",
            f"Удалённый доступ: {'да' if self.has_remote_access else 'нет'}",
            f"Подрядчики: {'да' if self.has_contractors else 'нет'}",
            f"Внешние интеграции: {'да' if self.has_external_integrations else 'нет'}",
            f"Облако/ЦОД: {self.cloud_model}",
            f"СЗИ/СКЗИ: {'; '.join(self.security_tools) or '—'}",
            f"Применяются СКЗИ (криптографическая защита): {'да' if self.has_crypto else 'нет'}",
            f"Особенности: {self.architecture_notes or self.notes or '—'}",
        ]
        return "\n".join(parts)


# ----------------------------------------------------------------------
INTERFACE_FLAG_MAP = {
    "Интернет": "has_internet",
    "Удаленный доступ": "has_remote_access",
    "Wi-Fi/беспроводные сети": "has_wireless",
    "API (внешние интеграции)": "has_external_integrations",
}


def sync_interface_flags(profile: Profile) -> Profile:
    """Выставляет булевы флаги по отмеченным network_interfaces (OR-логика:
    флаг становится True, если соответствующий интерфейс выбран; ручное
    включение флага без интерфейса сохраняется)."""
    selected = set(profile.network_interfaces or [])
    for label, flag in INTERFACE_FLAG_MAP.items():
        if label in selected:
            setattr(profile, flag, True)
    return profile


# ----------------------------------------------------------------------
STAGE_ORDER = [
    "s1_general",
    "s2_description",
    "s3_consequences",
    "s4_impact_objects",
    "s5_intruders",
    "s6_ways",
    "s7_ubi",
    "s8_scenarios",
    "s9_docx",
]

STAGE_TITLES = {
    "s1_general": "Раздел 1. Общие положения",
    "s2_description": "Раздел 2. Описание системы",
    "s3_consequences": "Раздел 3. Негативные последствия",
    "s4_impact_objects": "Раздел 4. Объекты воздействия",
    "s5_intruders": "Раздел 5. Источники угроз (нарушители)",
    "s6_ways": "Раздел 6. Способы реализации угроз",
    "s7_ubi": "Раздел 7.1. Анализ актуальных УБИ",
    "s8_scenarios": "Раздел 7.2. Сценарии (УБИ × тактики)",
    "s9_docx": "Сборка документа DOCX",
}


@dataclass
class JobState:
    """Состояние задания генерации (сохраняется в state.json)."""
    job_id: str = ""
    profile_name: str = ""
    profile_fingerprint: str = ""
    status: str = "new"          # new | running | paused | error | cancelled | done
    current_stage: str = ""
    error: str = ""
    stages: Dict[str, str] = field(default_factory=dict)   # stage -> pending|running|done|error
    created_at: str = ""
    updated_at: str = ""
    docx_path: str = ""

    @staticmethod
    def new(job_id: str, profile: Profile, profile_name: str) -> "JobState":
        now = datetime.now().isoformat(timespec="seconds")
        return JobState(
            job_id=job_id,
            profile_name=profile_name,
            profile_fingerprint=profile.fingerprint(),
            status="new",
            stages={s: "pending" for s in STAGE_ORDER},
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "JobState":
        known = {f for f in JobState.__dataclass_fields__}
        clean = {k: v for k, v in (d or {}).items() if k in known}
        js = JobState(**clean)
        for s in STAGE_ORDER:
            js.stages.setdefault(s, "pending")
        return js
