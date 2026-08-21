# -*- coding: utf-8 -*-
"""Веб-интерфейс сервиса генерации моделей угроз (Gradio).

Вкладки:
  1. Профиль ИС   — ввод исходных данных, сохранение/загрузка профилей,
                    автосохранение черновика (gr.Timer);
  2. Настройки LLM — ДВЕ конфигурации: LLM для анализа УБИ (JSON-задачи)
                    и LLM для текстов разделов/сценариев;
  3. Генерация     — запуск/возобновление/остановка, журнал хода работы;
  4. Результат     — скачивание DOCX и промежуточных данных.

Генерация выполняется в фоновом потоке; состояние опрашивается gr.Timer,
поэтому интерфейс не блокируется и переживает обновление страницы.
Черновик профиля, списки профилей/заданий и настройки LLM восстанавливаются
при загрузке страницы через demo.load.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

from tm_core import dictionaries as dicts, storage
from tm_core.pipeline import GenerationJob, start_or_resume
from tm_core.providers import LLMConfig, PROVIDER_LABELS, create_provider, LLMError
from tm_core.schema import (INTERFACE_FLAG_MAP, Component, Profile,
                            STAGE_ORDER, STAGE_TITLES, sync_interface_flags)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

# --------------------------- реестр активных заданий -------------------
ACTIVE_JOBS: Dict[str, GenerationJob] = {}
JOB_LOGS: Dict[str, List[str]] = {}
_jobs_lock = threading.Lock()

MISC = dicts.misc()
DMG = dicts.damages()["damage_types"]
INTRUDERS = dicts.intruders()["types"]

ALL_CONSEQUENCES = {d["id"]: d["consequences"] for d in DMG}
INTRUDER_CHOICES = [f"{t['id']}. {t['name']} [{t['category']}, {t['level']}] — {t['hint']}"
                    for t in INTRUDERS]
SECURITY_TOOLS_CHOICES = dicts.security_tools_choices()


def _intruder_ids_from_choices(choices: List[str]) -> List[int]:
    return [int(c.split(".", 1)[0]) for c in choices or []]


def _choices_from_intruder_ids(ids: List[int]) -> List[str]:
    return [c for c in INTRUDER_CHOICES if int(c.split(".", 1)[0]) in set(ids or [])]


# ======================================================================
# Профиль: сборка из полей UI и обратно
# ======================================================================
PROFILE_FIELDS = [
    "object_name", "operator_name", "responsible", "developer_org",
    "approver_position", "city", "year",
    "system_type", "protection_level", "classification_basis",
    "purpose", "business_processes", "scale",
    "info_kinds", "pdn_categories", "pdn_subjects", "pdn_volume",
    "components_df", "network_interfaces",
    "has_internet", "has_wireless", "has_remote_access",
    "has_contractors", "has_external_integrations",
    "cloud_model", "cloud_details", "security_tools", "has_crypto", "architecture_notes",
    "user_groups", "users_notes",
    "damage_types", "consequences", "consequences_custom",
    "intruder_choices", "intruders_excluded_reason", "notes",
]


def build_profile(*vals) -> Profile:
    d = dict(zip(PROFILE_FIELDS, vals))
    comps: List[Component] = []
    df = d.pop("components_df")
    if df is not None:
        try:
            records = df.to_dict("records") if hasattr(df, "to_dict") else list(df)
        except Exception:
            records = []
        for r in records:
            vals_list = list(r.values()) if isinstance(r, dict) else list(r)
            vals_list = [str(v).strip() if v is not None else "" for v in vals_list]
            if len(vals_list) >= 4 and any(vals_list[:4]):
                if vals_list[0] or vals_list[1]:
                    comps.append(Component(name=vals_list[0], ctype=vals_list[1],
                                           purpose=vals_list[2], location=vals_list[3]))
    intr = _intruder_ids_from_choices(d.pop("intruder_choices"))
    p = Profile.from_dict({k: v for k, v in d.items() if k in Profile.__dataclass_fields__})
    p.components = comps
    p.intruder_ids = intr
    p.year = str(d.get("year") or "").strip()
    sync_interface_flags(p)
    return p


def profile_to_ui(p: Profile) -> list:
    comp_rows = [[c.name, c.ctype, c.purpose, c.location] for c in p.components] or [["", "", "", ""]]
    return [
        p.object_name, p.operator_name, p.responsible, p.developer_org,
        p.approver_position, p.city, p.year,
        p.system_type or None, p.protection_level or None, p.classification_basis,
        p.purpose, p.business_processes, p.scale,
        p.info_kinds, p.pdn_categories, p.pdn_subjects, p.pdn_volume or None,
        comp_rows, p.network_interfaces,
        p.has_internet, p.has_wireless, p.has_remote_access,
        p.has_contractors, p.has_external_integrations,
        p.cloud_model or "Не используется", p.cloud_details, p.security_tools, p.has_crypto, p.architecture_notes,
        p.user_groups, p.users_notes,
        p.damage_types, p.consequences, p.consequences_custom,
        _choices_from_intruder_ids(p.intruder_ids), p.intruders_excluded_reason, p.notes,
    ]


# ======================================================================
# Обработчики: профиль
# ======================================================================

COMPONENT_HEADERS = ["Наименование", "Тип", "Назначение", "Размещение"]


def _df_to_rows(df_value):
    """Dataframe (pandas) или список -> чистый список непустых строк."""
    rows = df_value.values.tolist() if hasattr(df_value, "values") else (df_value or [])
    return [r for r in rows if any(str(c).strip() for c in r)]


def _rows_to_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COMPONENT_HEADERS)


def on_add_component(name, ctype, purpose, place, df_value):
    if not str(name).strip():
        return gr.DataFrame(value=_rows_to_df(_df_to_rows(df_value))), \
               "", "", "", "⚠️ Укажите хотя бы наименование компонента."
    rows = _df_to_rows(df_value)
    rows.append([name, ctype, purpose, place])
    return gr.DataFrame(value=_rows_to_df(rows)), \
           "", "", "", f"✅ Добавлен компонент: **{name}** (всего: {len(rows)})"


def on_remove_last_component(df_value):
    rows = _df_to_rows(df_value)
    if not rows:
        return gr.DataFrame(value=_rows_to_df(rows)), "ℹ️ Таблица уже пуста."
    removed = rows.pop()
    return gr.DataFrame(value=_rows_to_df(rows)), \
           f"🗑 Удалён компонент: **{removed[0]}** (осталось: {len(rows)})"


def on_save_profile(profile_name, *vals):
    p = build_profile(*vals)
    errors = p.validate()
    name = (profile_name or p.object_name or "").strip()
    if not name:
        return gr.update(), "❌ Укажите имя профиля или наименование объекта."
    path = storage.save_profile(p, name)
    msg = f"✅ Профиль сохранён: {Path(path).name}"
    if errors:
        msg += "\n⚠️ Профиль неполный:\n" + "\n".join(f"• {e}" for e in errors)
    return gr.update(choices=storage.list_profiles(), value=str(path)), msg


def on_load_profile(path):
    if not path:
        return [gr.update()] * len(PROFILE_FIELDS) + ["❌ Выберите профиль."]
    try:
        p = storage.load_profile(path)
    except Exception as e:
        return [gr.update()] * len(PROFILE_FIELDS) + [f"❌ Ошибка загрузки: {e}"]
    return profile_to_ui(p) + [f"✅ Загружен профиль: {Path(path).name}"]


def on_upload_profile(file):
    if file is None:
        return [gr.update()] * len(PROFILE_FIELDS) + ["❌ Файл не выбран."]
    try:
        p = storage.load_profile(file)
    except Exception as e:
        return [gr.update()] * len(PROFILE_FIELDS) + [f"❌ Ошибка чтения файла: {e}"]
    return profile_to_ui(p) + ["✅ Профиль загружен из файла."]


def on_refresh_profiles():
    return gr.update(choices=storage.list_profiles())


def on_damage_change(damage_types):
    cons = []
    for d in damage_types or []:
        cons.extend(ALL_CONSEQUENCES.get(d, []))
    return gr.update(choices=cons)


def on_security_tools_change(choices, has_crypto_val):
    """Автовключение чекбокса «Применяются СКЗИ» при выборе crypto-средства
    из справочника (автоматически чекбокс НЕ выключается)."""
    if has_crypto_val:
        return gr.update()
    if any(dicts.is_crypto_tool(c) for c in choices or []):
        return gr.update(value=True)
    return gr.update()


def on_interfaces_change(selected):
    """Автовключение булевых флагов по отмеченным интерфейсам/каналам
    (автоматически флаги НЕ выключаются — ручные включения сохраняются).

    Порядок outputs: has_internet, has_remote_access, has_wireless,
    has_external_integrations.
    """
    selected = set(selected or [])
    updates = {flag: gr.update() for flag in INTERFACE_FLAG_MAP.values()}
    for label, flag in INTERFACE_FLAG_MAP.items():
        if label in selected:
            updates[flag] = gr.update(value=True)
    return [updates["has_internet"], updates["has_remote_access"],
            updates["has_wireless"], updates["has_external_integrations"]]


def on_autosave_draft(*vals):
    """Автосохранение черновика профиля (вызывается по gr.Timer).

    Ошибки молча игнорируются — черновик не должен мешать работе.
    """
    try:
        p = build_profile(*vals)
        storage.save_draft(p.to_dict())
    except Exception:
        pass


# ======================================================================
# Обработчики: настройки LLM (две конфигурации: анализ и тексты)
# ======================================================================
PROVIDER_KEYS = list(PROVIDER_LABELS.keys())
PROVIDER_CHOICES = [(v, k) for k, v in PROVIDER_LABELS.items()]

DEFAULTS = {
    "ollama": ("http://localhost:11434", "qwen2.5:14b-instruct"),
    "openai": ("http://localhost:1234/v1", "local-model"),
    "anthropic": ("", "claude-sonnet-4-5"),
    "transformers": ("", "HuggingFaceH4/zephyr-7b-beta"),
    "mock": ("", "mock"),
}


def on_provider_change(provider):
    base, model = DEFAULTS.get(provider, ("", ""))
    need_key = provider == "anthropic"
    need_url = provider in ("ollama", "openai")
    return (
        gr.update(value=base, visible=need_url or provider == "anthropic"),
        gr.update(value=model),
        gr.update(visible=need_key or provider == "openai"),
    )


def _cfg_from_ui(provider, base_url, model, api_key, temperature, max_tokens, timeout) -> Dict[str, Any]:
    return LLMConfig(
        provider=provider, base_url=(base_url or "").strip(), model=(model or "").strip(),
        api_key=(api_key or "").strip(), temperature=float(temperature),
        max_tokens=int(max_tokens), timeout=int(timeout),
    ).to_dict()


def on_save_llm(*vals):
    """Сохраняет ОБЕ конфигурации: первые 7 полей — анализ, следующие — тексты."""
    cfg_analysis = _cfg_from_ui(*vals[:7])
    cfg_text = _cfg_from_ui(*vals[7:14])
    storage.save_llm_settings({"analysis": cfg_analysis, "text": cfg_text})
    return "✅ Настройки сохранены (файл output/llm_settings.json)"


def on_test_llm(provider, base_url, model, api_key, temperature, max_tokens, timeout):
    cfg = LLMConfig.from_dict(_cfg_from_ui(provider, base_url, model, api_key, temperature, max_tokens, timeout))
    try:
        prov = create_provider(cfg)
        return "✅ " + prov.test_connection()
    except LLMError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ {type(e).__name__}: {e}"


def _llm_cfg_pair() -> Optional[Dict[str, Any]]:
    """Возвращает {"analysis":..., "text":...} из сохранённых настроек или None."""
    settings = storage.load_llm_settings()
    cfg_analysis = settings.get("analysis")
    if not cfg_analysis:
        return None
    return {"analysis": cfg_analysis, "text": settings.get("text") or cfg_analysis}


# ======================================================================
# Обработчики: генерация
# ======================================================================
def _job_choices():
    jobs = storage.list_jobs()
    return [(f"{j.job_id} — {j.profile_name or j.job_id} [{j.status}]", j.job_id) for j in jobs]


def _append_log(job_id: str, stage: str, msg: str):
    with _jobs_lock:
        JOB_LOGS.setdefault(job_id, []).append(f"[{STAGE_TITLES.get(stage, stage)}] {msg}")
        JOB_LOGS[job_id] = JOB_LOGS[job_id][-400:]


def _run_in_thread(job: GenerationJob):
    with _jobs_lock:
        ACTIVE_JOBS[job.job_id] = job
    try:
        job.run()
    finally:
        with _jobs_lock:
            ACTIVE_JOBS.pop(job.job_id, None)


def on_start_generation(profile_name, ubi_full_scan, ubi_batch, *vals):
    p = build_profile(*vals)
    errors = p.validate()
    if errors:
        return gr.update(), "❌ Профиль не готов к генерации:\n" + "\n".join(f"• {e}" for e in errors)
    cfgs = _llm_cfg_pair()
    if not cfgs:
        return gr.update(), "❌ Не сохранены настройки LLM (вкладка «Настройки LLM»)."
    state = storage.new_job(p, (profile_name or p.object_name).strip())
    job = start_or_resume(
        state.job_id, cfgs["analysis"], cfgs["text"],
        progress=lambda s, m: _append_log(state.job_id, s, m),
        ubi_full_scan=bool(ubi_full_scan), ubi_batch_size=int(ubi_batch),
    )
    threading.Thread(target=_run_in_thread, args=(job,), daemon=True).start()
    return (gr.update(choices=_job_choices(), value=state.job_id),
            f"🚀 Задание {state.job_id} запущено.")


def on_resume_job(job_id, ubi_full_scan, ubi_batch):
    if not job_id:
        return "❌ Выберите задание."
    with _jobs_lock:
        if job_id in ACTIVE_JOBS:
            return "⚠️ Задание уже выполняется."
    st = storage.load_job_state(job_id)
    if st is None:
        return "❌ Задание не найдено."
    if st.status == "done":
        return "✅ Задание уже завершено — документ доступен на вкладке «Результат»."
    cfgs = _llm_cfg_pair()
    if not cfgs:
        return "❌ Не сохранены настройки LLM."
    job = start_or_resume(
        job_id, cfgs["analysis"], cfgs["text"],
        progress=lambda s, m: _append_log(job_id, s, m),
        ubi_full_scan=bool(ubi_full_scan), ubi_batch_size=int(ubi_batch),
    )
    threading.Thread(target=_run_in_thread, args=(job,), daemon=True).start()
    return f"▶️ Задание {job_id} возобновлено (готовые этапы и батчи пропускаются)."


def on_stop_job(job_id):
    with _jobs_lock:
        job = ACTIVE_JOBS.get(job_id)
    if job is None:
        return "⚠️ Задание не выполняется."
    job.cancel()
    return "⏸ Запрошена остановка — прогресс будет сохранён после текущего батча."


def on_poll(job_id):
    """Опрос состояния для gr.Timer."""
    if not job_id:
        return "—", "", gr.update()
    st = storage.load_job_state(job_id)
    if st is None:
        return "Задание не найдено", "", gr.update()
    lines = []
    for s in STAGE_ORDER:
        mark = {"done": "✅", "running": "⏳", "error": "❌", "pending": "▫️"}.get(st.stages.get(s, "pending"), "▫️")
        lines.append(f"{mark} {STAGE_TITLES.get(s, s)}")
    status_map = {"running": "Выполняется", "paused": "Приостановлено", "done": "Готово",
                  "error": "Ошибка", "new": "Создано", "cancelled": "Отменено"}
    head = f"Статус: {status_map.get(st.status, st.status)}"
    if st.error:
        head += f"\nОшибка: {st.error}"
    log = "\n".join(JOB_LOGS.get(job_id, [])[-60:])
    docx_update = gr.update(value=st.docx_path, visible=bool(st.docx_path)) if st.docx_path \
        else gr.update(visible=False)
    return head + "\n\n" + "\n".join(lines), log, docx_update


def on_refresh_jobs():
    return gr.update(choices=_job_choices())


def on_open_result(job_id):
    if not job_id:
        return gr.update(visible=False), gr.update(visible=False), "❌ Выберите задание."
    st = storage.load_job_state(job_id)
    if st is None:
        return gr.update(visible=False), gr.update(visible=False), "❌ Задание не найдено."
    files = []
    d = storage.job_dir(job_id)
    for f in sorted(d.glob("*.json")):
        files.append(str(f))
    docx = st.docx_path if st.docx_path and Path(st.docx_path).exists() else None
    msg = f"Задание {job_id}: статус «{st.status}»."
    if docx:
        msg += " Документ готов."
    return (gr.update(value=docx, visible=bool(docx)),
            gr.update(value=files, visible=bool(files)),
            msg)


# ======================================================================
# Загрузка страницы: черновик, списки, настройки LLM
# ======================================================================
def _llm_panel_updates(cfg: Dict[str, Any]) -> list:
    """Значения полей одной панели LLM из сохранённой конфигурации."""
    prov = cfg.get("provider", "ollama")
    base_default, model_default = DEFAULTS.get(prov, ("", ""))
    return [
        gr.update(value=prov),
        gr.update(value=cfg.get("base_url", base_default)),
        gr.update(value=cfg.get("model", model_default)),
        gr.update(value=cfg.get("api_key", "")),
        gr.update(value=cfg.get("temperature", 0.1)),
        gr.update(value=cfg.get("max_tokens", 4096)),
        gr.update(value=cfg.get("timeout", 600)),
    ]


def on_page_load():
    """demo.load: черновик профиля, списки профилей/заданий, настройки LLM."""
    draft = storage.load_draft()
    if draft:
        try:
            profile_vals = profile_to_ui(Profile.from_dict(draft))
        except Exception:
            profile_vals = [gr.update()] * len(PROFILE_FIELDS)
    else:
        profile_vals = [gr.update()] * len(PROFILE_FIELDS)
    settings = storage.load_llm_settings()
    llm_vals = (_llm_panel_updates(settings.get("analysis") or {})
                + _llm_panel_updates(settings.get("text") or {}))
    return (profile_vals
            + [gr.update(choices=storage.list_profiles()),
               gr.update(choices=_job_choices()),
               gr.update(choices=_job_choices())]
            + llm_vals)


# ======================================================================
# UI (styled)
# ======================================================================

# ---------- Тема ------------------------------------------------------
def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1",
            c400="#94a3b8", c500="#64748b", c600="#475569", c700="#334155",
            c800="#1e293b", c900="#0f172a", c950="#020617",
        ),
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )
    tweaks = {
        "body_background_fill": "#f4f6f9",
        "body_background_fill_dark": "#f4f6f9",
        "block_background_fill": "#ffffff",
        "block_background_fill_dark": "#ffffff",
        "block_border_width": "1px",
        "block_border_color": "#e2e8f0",
        "block_shadow": "0 1px 3px rgba(15, 23, 42, 0.06)",
        "block_radius": "10px",
        "block_title_text_weight": "600",
        "button_primary_background_fill": "#334155",
        "button_primary_background_fill_hover": "#1e293b",
        "button_primary_text_color": "#ffffff",
        "button_primary_border_width": "0px",
        "button_secondary_background_fill": "#ffffff",
        "button_secondary_background_fill_hover": "#f1f5f9",
        "button_secondary_border_color": "#cbd5e1",
        "button_secondary_text_color": "#334155",
        "input_background_fill": "#ffffff",
        "input_background_fill_dark": "#ffffff",
        "input_border_color": "#d3dae3",
        "input_border_color_focus": "#64748b",
        "accordion_background_fill": "#ffffff",
        "border_color_accent_subdued": "#e2e8f0",
        "body_text_color": "#1e293b",
        "body_text_color_subdued": "#64748b",
    }
    for key, value in tweaks.items():
        try:
            theme = theme.set(**{key: value})
        except (TypeError, ValueError):
            pass
    return theme


CUSTOM_CSS = """
/* ---------- Шапка ---------- */
.app-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #334155;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
}
.app-header h1 {
    color: #0f172a !important;
    margin: 0 0 8px 0 !important;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.app-header p {
    color: #64748b !important;
    margin: 0 !important;
    font-size: 0.93rem;
    line-height: 1.55;
}

/* ---------- Вкладки ---------- */
.tab-nav button {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    color: #475569 !important;
}
.tab-nav button.selected {
    background: #334155 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* ---------- Аккордеоны как карточки ---------- */
.accordion-card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-bottom: 14px !important;
    overflow: hidden;
    background: #ffffff !important;
}
.accordion-card > .label-wrap {
    background: #f8fafc !important;
    border-bottom: 1px solid #eef2f6 !important;
    padding: 10px 16px !important;
    font-weight: 600;
    color: #334155;
}

/* ---------- Бейдж обязательных полей ---------- */
.req::after {
    content: " *";
    color: #dc2626;
    font-weight: 700;
}

/* ---------- Кнопка запуска ---------- */
.launch-btn button {
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    padding: 13px 0 !important;
    border-radius: 10px !important;
}

/* ---------- Логи ---------- */
.logbox textarea {
    font-family: "JetBrains Mono", monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
    background: #f8fafc !important;
    color: #334155 !important;
}
.logbox textarea::-webkit-scrollbar { width: 8px; }
.logbox textarea::-webkit-scrollbar-thumb {
    background: #cbd5e1; border-radius: 4px;
}

/* ---------- Подсказки ---------- */
.hint {
    background: #f8fafc;
    border-left: 4px solid #94a3b8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    color: #475569;
    font-size: 0.9rem;
}

/* ---------- Основные кнопки ---------- */
button.primary, .gr-button-primary {
    border: none !important;
    background: #334155 !important;
    color: #ffffff !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #1e293b !important;
}

/* ---------- Поля ввода: контраст на белой карточке ---------- */
.gr-textbox textarea, .gr-textbox input,
textarea, input[type="text"], input[type="password"], input[type="number"] {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
textarea:focus, input[type="text"]:focus,
input[type="password"]:focus, input[type="number"]:focus {
    background: #ffffff !important;
    border-color: #64748b !important;
    box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.18) !important;
    outline: none !important;
}

/* Выпадающие списки */
.gr-dropdown, .dropdown, select {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
"""


# ---------- Приложение ------------------------------------------------
def build_app() -> gr.Blocks:
    storage.ensure_dirs()
    with gr.Blocks(
        title="Генератор моделей угроз (ФСТЭК)",
        theme=build_theme(),
        css=CUSTOM_CSS,
    ) as demo:

        gr.HTML("""
        <div class="app-header">
            <h1>🛡️ Генератор моделей угроз безопасности информации</h1>
            <p>Автоматическое формирование документа по Методике ФСТЭК России
               от 05.02.2021 и приказу ФСТЭК России № 21.
               Поддерживаются локальные LLM (Ollama, LM Studio, vLLM)
               и облачный Anthropic Claude. Для анализа УБИ и для текстов
               разделов можно настроить разные модели.</p>
        </div>
        """)

        # -------------------------------------------- Вкладка 1: профиль
        with gr.Tab("📋 1. Профиль ИС"):
            with gr.Row():
                profile_dd = gr.Dropdown(label="Сохранённые профили",
                                         choices=storage.list_profiles(), scale=3)
                load_btn = gr.Button("📂 Загрузить", scale=1)
                refresh_btn = gr.Button("🔄 Обновить список", scale=1)
            with gr.Row():
                profile_name = gr.Textbox(label="Имя профиля (для сохранения)", scale=3)
                save_btn = gr.Button("💾 Сохранить профиль", variant="primary", scale=1)
                upload = gr.File(label="Импорт профиля (.json)", file_types=[".json"], scale=2)
            profile_status = gr.Markdown()

            with gr.Accordion("📄 Реквизиты документа", open=True,
                              elem_classes=["accordion-card"]):
                with gr.Row():
                    object_name = gr.Textbox(label="Наименование ИС",
                                             elem_classes=["req"])
                    operator_name = gr.Textbox(label="Оператор (обладатель информации)",
                                               elem_classes=["req"])
                with gr.Row():
                    responsible = gr.Textbox(label="Ответственное подразделение/лицо")
                    developer_org = gr.Textbox(label="Разработчик модели угроз")
                with gr.Row():
                    approver_position = gr.Textbox(label="Должность утверждающего")
                    city = gr.Textbox(label="Город")
                    year = gr.Textbox(label="Год", value="2026")

            with gr.Accordion("🏷️ Классификация и назначение", open=True,
                              elem_classes=["accordion-card"]):
                with gr.Row():
                    system_type = gr.Dropdown(label="Тип системы",
                                              choices=MISC["system_types"],
                                              elem_classes=["req"])
                    protection_level = gr.Dropdown(label="УЗ ПДн / класс ГИС",
                                                   choices=MISC["protection_levels"])
                    classification_basis = gr.Textbox(label="Реквизиты акта классификации")
                purpose = gr.Textbox(label="Назначение и задачи ИС", lines=2)
                business_processes = gr.Textbox(label="Основные (бизнес-)процессы", lines=2)
                scale = gr.Textbox(label="Масштаб и размещение (площадки, адреса)", lines=2)

            with gr.Accordion("🗃️ Обрабатываемая информация", open=True,
                              elem_classes=["accordion-card"]):
                info_kinds = gr.CheckboxGroup(label="Виды информации",
                                              choices=MISC["info_kinds"],
                                              elem_classes=["req"])
                with gr.Row():
                    pdn_categories = gr.CheckboxGroup(label="Категории ПДн",
                                                      choices=MISC["pdn_categories"])
                    pdn_subjects = gr.CheckboxGroup(label="Субъекты ПДн",
                                                    choices=MISC["pdn_subjects"])
                    pdn_volume = gr.Dropdown(label="Объём субъектов ПДн",
                                             choices=MISC["pdn_volume"])

            with gr.Accordion("🏗️ Архитектура", open=True,
                              elem_classes=["accordion-card"]):

                # --- Быстрое добавление компонента ---
                gr.Markdown("**Добавить компонент ИС** — заполните поля и нажмите «➕».")
                with gr.Row():
                    comp_name = gr.Textbox(label="Наименование", scale=3,
                                           placeholder="Например: Сервер БД")
                    comp_type = gr.Dropdown(label="Тип", choices=MISC["component_types"],
                                            allow_custom_value=True, scale=2)
                with gr.Row():
                    comp_purpose = gr.Textbox(label="Назначение", scale=3,
                                              placeholder="Хранение и обработка ПДн")
                    comp_place = gr.Textbox(label="Размещение", scale=2,
                                            placeholder="Серверная, 2 этаж")
                with gr.Row():
                    comp_add_btn = gr.Button("➕ Добавить компонент", variant="primary", scale=1)
                    comp_del_btn = gr.Button("🗑 Удалить последнюю строку", scale=1)
                comp_status = gr.Markdown()

                # --- Таблица: просмотр и точечная правка ---
                components_df = gr.Dataframe(
                    label="Компоненты ИС (объекты воздействия)",
                    headers=["Наименование", "Тип", "Назначение", "Размещение"],
                    datatype=["str", "str", "str", "str"],
                    row_count=(1, "dynamic"), col_count=(4, "fixed"),
                    interactive=True, wrap=True,
                    elem_classes=["big-table", "req"],
                )
                gr.HTML(
                    "<div class='hint'>💡 Таблица выше — это список добавленных компонентов. "
                    "Исправить любую ячейку можно прямо в ней (клик по ячейке). "
                    "Заполнять таблицу вручную построчно не обязательно — используйте форму выше.</div>"
                )

                network_interfaces = gr.CheckboxGroup(label="Интерфейсы и каналы",
                                                      choices=MISC["network_interfaces"])
                gr.Markdown(
                    "Отметки \"Интернет\", \"Удаленный доступ\", \"Wi-Fi\" и \"API\" "
                    "автоматически включают соответствующие флаги ниже — они "
                    "учитываются при оценке актуальности УБИ",
                    elem_classes=["hint"],
                )
                with gr.Row():
                    has_internet = gr.Checkbox(label="Подключение к Интернет")
                    has_wireless = gr.Checkbox(label="Беспроводные сети")
                    has_remote_access = gr.Checkbox(label="Удалённый доступ")
                    has_contractors = gr.Checkbox(label="Подрядчики")
                    has_external_integrations = gr.Checkbox(label="Внешние интеграции")
                with gr.Row():
                    cloud_model = gr.Dropdown(label="ЦОД/облако",
                                              choices=MISC["cloud_models"],
                                              value="Не используется")
                    cloud_details = gr.Textbox(
                        label="Детали размещения / разграничение ответственности")
                security_tools = gr.Dropdown(
                    label="Применяемые СЗИ/СКЗИ",
                    choices=SECURITY_TOOLS_CHOICES,
                    multiselect=True, allow_custom_value=True)
                gr.HTML(
                    "<div class='hint'>💡 Выберите средства из справочника или "
                    "впишите своё (введите название и нажмите Enter). Если выбрано "
                    "средство криптографической защиты (СКЗИ), чекбокс ниже "
                    "включается автоматически.</div>"
                )
                has_crypto = gr.Checkbox(
                    label="Применяются СКЗИ (криптографическая защита)")
                architecture_notes = gr.Textbox(label="Дополнительно об архитектуре", lines=3)

            with gr.Accordion("👥 Пользователи", open=False,
                              elem_classes=["accordion-card"]):
                user_groups = gr.CheckboxGroup(label="Группы пользователей",
                                               choices=MISC["user_groups"])
                users_notes = gr.Textbox(label="Дополнительно о пользователях", lines=2)

            with gr.Accordion("⚠️ Негативные последствия (раздел 3)", open=True,
                              elem_classes=["accordion-card"]):
                damage_types = gr.CheckboxGroup(
                    label="Виды риска (ущерба)",
                    choices=[(f"{d['id']} — {d['name']}", d["id"]) for d in DMG],
                    elem_classes=["req"],
                )
                consequences = gr.CheckboxGroup(
                    label="Типовые негативные последствия (по табл. 4.1)", choices=[])
                consequences_custom = gr.Textbox(
                    label="Свои последствия (по одному в строке)", lines=3)

            with gr.Accordion("🕵️ Нарушители (раздел 5)", open=True,
                              elem_classes=["accordion-card"]):
                intruder_choices = gr.CheckboxGroup(label="Виды нарушителей",
                                                    choices=INTRUDER_CHOICES,
                                                    elem_classes=["req"])
                intruders_excluded_reason = gr.Textbox(
                    label="Обоснование исключения остальных видов", lines=2)

            notes = gr.Textbox(label="Прочие примечания", lines=2)

            gr.HTML(
                "<div class='hint'>💾 Черновик профиля автоматически сохраняется "
                "каждые 10 секунд — при обновлении страницы введённые данные "
                "восстановятся.</div>"
            )

        profile_inputs = [
            object_name, operator_name, responsible, developer_org,
            approver_position, city, year,
            system_type, protection_level, classification_basis,
            purpose, business_processes, scale,
            info_kinds, pdn_categories, pdn_subjects, pdn_volume,
            components_df, network_interfaces,
            has_internet, has_wireless, has_remote_access,
            has_contractors, has_external_integrations,
            cloud_model, cloud_details, security_tools, has_crypto, architecture_notes,
            user_groups, users_notes,
            damage_types, consequences, consequences_custom,
            intruder_choices, intruders_excluded_reason, notes,
        ]

        damage_types.change(on_damage_change, [damage_types], [consequences])
        security_tools.change(on_security_tools_change,
                              [security_tools, has_crypto], [has_crypto])
        network_interfaces.change(
            on_interfaces_change, [network_interfaces],
            [has_internet, has_remote_access, has_wireless,
             has_external_integrations])
        save_btn.click(on_save_profile, [profile_name] + profile_inputs,
                       [profile_dd, profile_status])
        load_btn.click(on_load_profile, [profile_dd], profile_inputs + [profile_status])
        upload.upload(on_upload_profile, [upload], profile_inputs + [profile_status])
        refresh_btn.click(on_refresh_profiles, None, [profile_dd])
        comp_add_btn.click(
            on_add_component,
            [comp_name, comp_type, comp_purpose, comp_place, components_df],
            [components_df, comp_name, comp_type, comp_purpose, comp_status],
        )
        comp_del_btn.click(
            on_remove_last_component,
            [components_df],
            [components_df, comp_status],
        )

        # автосохранение черновика профиля (тихое, раз в 10 секунд)
        draft_timer = gr.Timer(10)
        draft_timer.tick(on_autosave_draft, profile_inputs)

        # -------------------------------------------- Вкладка 2: LLM
        with gr.Tab("🤖 2. Настройки LLM"):
            gr.HTML(
                "<div class='hint'>Можно задать <b>две разные модели</b>: для "
                "аналитических JSON-задач (оценка УБИ и тактик) и для генерации "
                "связных текстов разделов. Если конфигурации одинаковые — просто "
                "заполните обе одинаково.</div>"
            )

            with gr.Accordion("🔍 LLM для анализа УБИ (разделы 7.1–7.2)", open=True,
                              elem_classes=["accordion-card"]):
                gr.HTML(
                    "<div class='hint'>Здесь важна <b>дисциплина JSON</b>: модель "
                    "должна стабильно отвечать валидным JSON-массивом. "
                    "Параметр «Максимум токенов ответа» — не менее <b>4096</b>, "
                    "иначе ответ на батч УБИ обрезается.</div>"
                )
                a_provider = gr.Radio(label="Провайдер", choices=PROVIDER_CHOICES,
                                      value="ollama")
                a_base_url = gr.Textbox(label="Адрес сервера (base URL)",
                                        value=DEFAULTS["ollama"][0])
                a_model = gr.Textbox(label="Модель", value=DEFAULTS["ollama"][1])
                a_api_key = gr.Textbox(label="API-ключ (если требуется)", type="password")
                with gr.Row():
                    a_temperature = gr.Slider(0.0, 1.0, value=0.1,
                                              step=0.05, label="Temperature")
                    a_max_tokens = gr.Slider(256, 16384, value=4096,
                                             step=256, label="Максимум токенов ответа")
                    a_timeout = gr.Slider(30, 3600, value=600,
                                          step=30, label="Таймаут запроса, сек")
                a_test_btn = gr.Button("🔌 Проверить подключение")
                a_status = gr.Markdown()

            with gr.Accordion("✍️ LLM для текстов разделов (разделы 1–6, сценарии)", open=True,
                              elem_classes=["accordion-card"]):
                gr.HTML(
                    "<div class='hint'>Здесь нужны <b>длинные связные тексты</b> — "
                    "желательно большое контекстное окно (<b>16k+</b>). Сюда можно "
                    "указать более сильную внешнюю/облачную модель, даже если "
                    "анализ выполняется локально.</div>"
                )
                t_provider = gr.Radio(label="Провайдер", choices=PROVIDER_CHOICES,
                                      value="ollama")
                t_base_url = gr.Textbox(label="Адрес сервера (base URL)",
                                        value=DEFAULTS["ollama"][0])
                t_model = gr.Textbox(label="Модель", value=DEFAULTS["ollama"][1])
                t_api_key = gr.Textbox(label="API-ключ (если требуется)", type="password")
                with gr.Row():
                    t_temperature = gr.Slider(0.0, 1.0, value=0.2,
                                              step=0.05, label="Temperature")
                    t_max_tokens = gr.Slider(256, 16384, value=4096,
                                             step=256, label="Максимум токенов ответа")
                    t_timeout = gr.Slider(30, 3600, value=600,
                                          step=30, label="Таймаут запроса, сек")
                t_test_btn = gr.Button("🔌 Проверить подключение")
                t_status = gr.Markdown()

            save_llm_btn = gr.Button("💾 Сохранить обе конфигурации", variant="primary")
            llm_status = gr.Markdown()
            gr.HTML(
                "<div class='hint'>💡 <b>Подсказки.</b> "
                "Ollama: <code>http://localhost:11434</code>, модель — например "
                "<code>qwen2.5:14b-instruct</code>. "
                "LM Studio: <code>http://localhost:1234/v1</code>. "
                "vLLM: <code>http://host:8000/v1</code>. "
                "Claude: нужен API-ключ Anthropic. Провайдер «Mock» позволяет проверить "
                "весь конвейер и получить тестовый DOCX без LLM.</div>"
            )
            gr.HTML(
                "<div class='hint'>📏 <b>Требования к моделям.</b> "
                "Нужны instruct-модели с уверенным русским языком. "
                "<b>Для анализа УБИ:</b> стабильный JSON-вывод, max_tokens ≥ 4096, "
                "контекстное окно от 8k; если модель слабая — на вкладке «Генерация» "
                "установите «УБИ в одном запросе» = 1–2. "
                "<b>Для текстов разделов:</b> длинные связные тексты, контекстное окно "
                "16k+ (подходят внешние/облачные модели). "
                "Модели меньше 7B не рекомендуются ни для одной роли.</div>"
            )

            a_fields = [a_provider, a_base_url, a_model, a_api_key,
                        a_temperature, a_max_tokens, a_timeout]
            t_fields = [t_provider, t_base_url, t_model, t_api_key,
                        t_temperature, t_max_tokens, t_timeout]
            a_provider.change(on_provider_change, [a_provider],
                              [a_base_url, a_model, a_api_key])
            t_provider.change(on_provider_change, [t_provider],
                              [t_base_url, t_model, t_api_key])
            a_test_btn.click(on_test_llm, a_fields, [a_status])
            t_test_btn.click(on_test_llm, t_fields, [t_status])
            save_llm_btn.click(on_save_llm, a_fields + t_fields, [llm_status])

        # -------------------------------------------- Вкладка 3: генерация
        with gr.Tab("🚀 3. Генерация"):
            gr.HTML(
                "<div class='hint'>Генерация идёт по этапам с сохранением прогресса. "
                "Её можно остановить и возобновить позже — готовые этапы и "
                "обработанные батчи УБИ не пересчитываются.</div>"
            )
            with gr.Row():
                ubi_full_scan = gr.Checkbox(
                    label="Полный перебор всех УБИ (точнее, но дольше)", value=True)
                ubi_batch = gr.Slider(1, 10, value=4, step=1,
                                      label="УБИ в одном запросе к LLM")
            with gr.Row(elem_classes=["launch-btn"]):
                start_btn = gr.Button("🚀 Запустить генерацию", variant="primary")
            with gr.Row():
                job_dd = gr.Dropdown(label="Задания", choices=_job_choices(), scale=3)
                refresh_jobs_btn = gr.Button("🔄 Обновить", scale=1)
                resume_btn = gr.Button("▶️ Продолжить", scale=1)
                stop_btn = gr.Button("⏸ Остановить", scale=1)
            gen_status = gr.Markdown("—")
            with gr.Row():
                stages_md = gr.Textbox(label="Этапы", lines=12, interactive=False,
                                       elem_classes=["logbox"])
                log_md = gr.Textbox(label="Журнал", lines=12, interactive=False,
                                    elem_classes=["logbox"])
            docx_quick = gr.File(label="Готовый документ", visible=False)

            timer = gr.Timer(3.0)
            timer.tick(on_poll, [job_dd], [stages_md, log_md, docx_quick])

            start_btn.click(on_start_generation,
                            [profile_name, ubi_full_scan, ubi_batch] + profile_inputs,
                            [job_dd, gen_status])
            resume_btn.click(on_resume_job, [job_dd, ubi_full_scan, ubi_batch],
                             [gen_status])
            stop_btn.click(on_stop_job, [job_dd], [gen_status])
            refresh_jobs_btn.click(on_refresh_jobs, None, [job_dd])

        # -------------------------------------------- Вкладка 4: результат
        with gr.Tab("📦 4. Результат"):
            with gr.Row():
                res_job_dd = gr.Dropdown(label="Задание", choices=_job_choices(), scale=3)
                res_refresh = gr.Button("🔄 Обновить", scale=1)
                res_open = gr.Button("Показать результаты", variant="primary", scale=1)
            res_status = gr.Markdown()
            res_docx = gr.File(label="Модель угроз (DOCX)", visible=False)
            res_files = gr.File(label="Промежуточные данные этапов (JSON)",
                                visible=False, file_count="multiple")

            res_refresh.click(on_refresh_jobs, None, [res_job_dd])
            res_open.click(on_open_result, [res_job_dd],
                           [res_docx, res_files, res_status])

        # ------------------------------------ восстановление при F5
        demo.load(on_page_load, None,
                  profile_inputs + [profile_dd, job_dd, res_job_dd]
                  + a_fields + t_fields)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
