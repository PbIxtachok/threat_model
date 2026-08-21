# -*- coding: utf-8 -*-
"""LLM-анализ: применимость УБИ к объекту и сопоставление УБИ с тактиками.

Переработанные версии ubi_analysis.py и анализа тактик из app10.py:
  * провайдер LLM передаётся снаружи (любой из providers.py);
  * возобновление: обработанные элементы читаются из partial-чекпоинтов;
  * кэш ответов LLM со стабильными sha256-ключами (с версией промптов);
  * устойчивые JSON-вызовы: ретраи, ремонт ответов, поштучная дообработка;
  * промпты без двойного шаблонирования, json_mode у провайдера.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from . import dictionaries as dicts
from . import intruder_logic
from .providers import BaseProvider, safe_parse_json
from .retrieval import BM25
from .schema import Profile

logger = logging.getLogger(__name__)

# Версия промптов: входит во ВСЕ ключи кэша — старые (в т.ч. «отравленные»)
# записи кэша перестают совпадать и переиспользоваться.
PROMPT_VERSION = "v2"

# Число попыток LLM-запроса на батч (и на одиночную дообработку).
LLM_RETRIES = 3

ProgressCb = Callable[[str], None]   # сообщение для журнала/статуса
CancelCb = Callable[[], bool]        # True => остановиться


def _chunk(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _norm_conf(v: str) -> str:
    m = {"high": "высокая", "medium": "средняя", "low": "низкая",
         "высокая": "высокая", "средняя": "средняя", "низкая": "низкая"}
    return m.get(str(v or "").strip().lower(), "низкая")


def _norm_ubi_number(s: Any) -> str:
    """Нормализация номера УБИ: только цифры, без ведущих нулей.

    «УБИ.042» / «042» / «УБИ 42» → «42»; пусто/без цифр → «».
    """
    digits = "".join(re.findall(r"\d+", str(s or "")))
    return str(int(digits)) if digits else ""


def _norm_tactic_id(s: Any) -> str:
    """Нормализация кода тактики/техники: upper, strip, без точек на концах."""
    return str(s or "").strip().upper().strip(".")


# ======================================================================
# Устойчивый вызов LLM с JSON-ответом (ретраи + нормализация ключей)
# ======================================================================
def _collect_map(parsed: Any, key_normalizer: Callable[[Any], str]) -> Dict[str, Any]:
    """Строит map «нормализованный ключ → объект» из распарсенного JSON."""
    if isinstance(parsed, dict):
        parsed = parsed.get("results", [parsed])
    out: Dict[str, Any] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            raw_key = item.get("Number", item.get("tactic_id", ""))
            key = key_normalizer(raw_key)
            if key:
                out[key] = item
    return out


def call_llm_json(
    provider: BaseProvider,
    system: str,
    user: str,
    expected_keys: Iterable[str],
    *,
    key_normalizer: Callable[[Any], str],
    cache=None,
    cache_key: str = "",
    retries: int = LLM_RETRIES,
    cancelled: Optional[CancelCb] = None,
    progress: Optional[ProgressCb] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Устойчивый JSON-вызов LLM.

    Возвращает ``(parsed_map по нормализованным ключам, список failed_keys)``.

    Логика:
      * если ``cache_key`` задан — читаем кэш; валидный закэшированный ответ
        (парсится и покрывает ВСЕ ``expected_keys``) используется сразу;
      * до ``retries`` попыток ``generate()``; после каждой — ``safe_parse_json``
        и сбор map;
      * если часть ключей не получена, следующая попытка идёт с дополнением к
        ``user``: перечисляются недостающие ключи и требуется вернуть ТОЛЬКО их
        валидным JSON;
      * в кэш пишется ТОЛЬКО объединённый результат всех попыток, покрывающий
        ВСЕ ``expected_keys`` (JSON-список значений map); частичный сырой ответ
        последней попытки не кэшируется — он не прошёл бы проверку покрытия;
      * ``cancelled()`` между попытками — выход;
      * ``failed_keys`` — ключи, не полученные после всех попыток.
    """
    expected = [k for k in (key_normalizer(k) for k in expected_keys) if k]
    parsed_map: Dict[str, Any] = {}

    if cache is not None and cache_key:
        raw_cached = cache.get(cache_key)
        if raw_cached:
            cached_map = _collect_map(safe_parse_json(raw_cached), key_normalizer)
            if cached_map and all(k in cached_map for k in expected):
                return cached_map, []
            if progress:
                progress("Кэш: сохранённый результат не парсится или покрывает "
                         "не все ключи — запрос к LLM будет повторён.")

    user_prompt = user
    last_raw = ""
    for attempt in range(1, max(1, retries) + 1):
        if cancelled and cancelled():
            break
        try:
            last_raw = provider.generate(system, user_prompt, json_mode=True) or ""
        except Exception as e:
            logger.warning("Ошибка LLM (попытка %d/%d): %s", attempt, retries, e)
            last_raw = ""
            if progress:
                progress(f"Попытка {attempt}/{retries}: ошибка LLM ({e})")
        parsed_map.update(_collect_map(safe_parse_json(last_raw), key_normalizer))
        missing = [k for k in expected if k not in parsed_map]
        if not missing:
            if cache is not None and cache_key:
                # кэшируем ОБЪЕДИНЁННЫЙ результат всех попыток (JSON-список
                # значений map) — он гарантированно покрывает все ключи и
                # пройдёт проверку покрытия при чтении из кэша
                cache.put(cache_key, json.dumps(list(parsed_map.values()),
                                                ensure_ascii=False))
            return parsed_map, []
        if attempt < retries:
            user_prompt = (
                user + "\n\nВ предыдущем ответе отсутствовали результаты по "
                "элементам: " + ", ".join(missing) + ". Верни результат ТОЛЬКО "
                "по этим элементам — валидным JSON-массивом объектов того же "
                "формата, без пояснений вне JSON."
            )
            if progress:
                progress(f"Попытка {attempt}/{retries}: не получены ключи "
                         f"{', '.join(missing)} — повторный запрос.")
    failed = [k for k in expected if k not in parsed_map]
    return parsed_map, failed


# ======================================================================
# Расширенный контекст профиля для промптов
# ======================================================================
def profile_context(profile: Profile) -> str:
    """Расширенный текстовый контекст профиля для промптов LLM.

    ``profile.summary()`` + назначение, бизнес-процессы, масштаб, группы
    пользователей, выбранные виды ущерба и негативные последствия,
    нарушители (виды, категории, уровни Н1–Н4), СЗИ.
    """
    p = profile
    parts = [p.summary()]
    parts.append(f"Оператор (обладатель информации): {p.operator_name or '—'}")
    parts.append(f"Основные (бизнес-)процессы: {p.business_processes or '—'}")
    parts.append(f"Масштаб и размещение: {p.scale or '—'}")
    parts.append(f"Группы пользователей: {', '.join(p.user_groups) or '—'}")
    if p.users_notes:
        parts.append(f"Дополнительно о пользователях: {p.users_notes}")

    dmg = dicts.damages()["damage_types"]
    selected = [d for d in dmg if d["id"] in p.damage_types]
    if selected:
        parts.append("Выбранные виды ущерба: " +
                     "; ".join(f"{d['id']} — {d['name']}" for d in selected))
    if p.consequences:
        parts.append("Негативные последствия: " + "; ".join(p.consequences))
    custom = [s.strip() for s in p.consequences_custom.splitlines() if s.strip()]
    if custom:
        parts.append("Дополнительные последствия (определены оператором): " +
                     "; ".join(custom))

    if p.intruder_ids:
        lines = []
        for t in intruder_logic.selected_intruders(p.intruder_ids):
            lines.append(f"вид {t['id']} «{t['name']}» (категория: {t['category']}, "
                         f"уровень возможностей {t['level']})")
        parts.append("Актуальные нарушители: " + "; ".join(lines))
        lvl = intruder_logic.max_level(p.intruder_ids)
        info = intruder_logic.level_info(lvl)
        parts.append(f"Максимальный уровень возможностей нарушителей: {lvl} "
                     f"({info.get('title', '').lower()})")
    return "\n".join(parts)


# ======================================================================
# Rule-based префильтр УБИ по профилю
# ======================================================================
def rule_score_ubi(row: Dict[str, str], profile: Profile) -> Dict[str, Any]:
    text = f"{row['text']} {row['description']}".lower()
    score, reasons = 0, []
    excluded = False

    notes = " ".join([
        " ".join(profile.info_kinds), profile.architecture_notes, profile.notes,
        " ".join(profile.network_interfaces), profile.cloud_model,
    ]).lower()

    has_grid = "грид" in notes
    has_cloud = profile.cloud_model not in ("", "Не используется", "Собственный ЦОД")
    has_mobile = any("мобильн" in c.ctype.lower() for c in profile.components) or "мобильн" in notes
    has_wireless = profile.has_wireless
    has_virt = any("виртуализ" in (c.ctype + c.purpose).lower() for c in profile.components) or "виртуализ" in notes
    has_bigdata = "big data" in notes or "больших данных" in notes
    has_scada = any(k in (c.ctype.lower() + " " + c.purpose.lower()) for c in profile.components for k in ("плк", "scada")) or profile.system_type == "АСУ ТП"

    checks = [
        ("грид", ["грид"], has_grid, -100, "В составе ИС отсутствуют грид-системы"),
        ("облако", ["облачн"], has_cloud, -100, "Облачные технологии не применяются"),
        ("мобильные", ["мобильн"], has_mobile, -50, "Мобильные устройства не применяются"),
        ("беспроводные", ["беспровод", "wi-fi"], has_wireless, -50, "Беспроводные сети отсутствуют"),
        ("виртуализация", ["виртуальн", "гипервизор"], has_virt, -50, "Средства виртуализации не применяются"),
        ("big data", ["больших данных", "big data"], has_bigdata, -100, "Технологии больших данных не применяются"),
        ("АСУ ТП", ["суперкомпьютер"], False, -100, "Суперкомпьютеры не применяются"),
    ]
    for feature, needles, present, penalty, reason in checks:
        if any(n in text for n in needles):
            if present:
                score += 2
                reasons.append(f"Признак «{feature}» присутствует в профиле")
            else:
                score += penalty
                reasons.append(reason)
                if penalty <= -100:
                    excluded = True

    if any(n in text for n in ("сеть", "трафик", "vpn", "канал", "удален")):
        if profile.has_internet or profile.has_remote_access or profile.network_interfaces:
            score += 2
            reasons.append("Есть сетевые интерфейсы/каналы доступа")
    if any(n in text for n in ("интеграц", "доверенн", "обмен")):
        if profile.has_external_integrations:
            score += 2
            reasons.append("Есть внешние интеграции")
    if any(n in text for n in ("физическ", "bios", "uefi", "ремонт", "обслуживан")):
        if profile.has_contractors:
            score += 2
            reasons.append("Есть подрядчики/обслуживание")

    return {"rule_score": score, "rule_reasons": "; ".join(reasons), "excluded_by_rules": excluded}


# ======================================================================
# Этап: применимость УБИ (раздел 7.1)
# ======================================================================
UBI_SYSTEM_PROMPT = (
    "Ты — специалист по информационной безопасности. Оцениваешь применимость угроз "
    "безопасности информации (УБИ из БДУ ФСТЭК России) к конкретной информационной системе. "
    "Отвечаешь строго валидным JSON-массивом без пояснений вне JSON."
)

UBI_USER_TEMPLATE = """Профиль объекта:
{profile_summary}

Для КАЖДОЙ УБИ из списка ниже определи, применима ли она к данному объекту.
Опирайся на факты профиля и типовые свойства такой инфраструктуры. Если в профиле
явно отсутствует необходимая технология — matches=false.

Список УБИ:
{ubi_batch_text}

Ответ — JSON-массив объектов вида:
[{{"Number": "<номер УБИ — ТОЛЬКО цифры, как в списке (например, 42)>", "matches": true/false, "confidence": "высокая|средняя|низкая", "explanation": "1-2 предложения на русском"}}]
Верни по одному объекту на КАЖДУЮ УБИ из списка, без пояснений вне JSON.
"""


def _ubi_batch_text(batch: List[Dict[str, str]]) -> str:
    return "\n---\n".join(
        f"УБИ {r['Number']}\nНаименование: {r['text']}\nОписание: {r['description'][:800]}"
        for r in batch
    )


def analyze_ubi(
    profile: Profile,
    provider: BaseProvider,
    *,
    batch_size: int = 4,
    full_scan: bool = True,
    top_n: int = 40,
    done_numbers: Optional[Set[str]] = None,
    cache=None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress: Optional[ProgressCb] = None,
    cancelled: Optional[CancelCb] = None,
    retries: int = LLM_RETRIES,
) -> List[Dict[str, Any]]:
    """Возвращает список результатов по УБИ (только вновь обработанные)."""
    ubi_rows = dicts.load_ubi()
    scored = []
    for row in ubi_rows:
        r = dict(row)
        r.update(rule_score_ubi(row, profile))
        scored.append(r)

    excluded = [r for r in scored if r["excluded_by_rules"]]
    candidates = [r for r in scored if not r["excluded_by_rules"]]

    if progress:
        progress(f"УБИ всего: {len(scored)}; исключено правилами: {len(excluded)}; кандидатов: {len(candidates)}")

    if not full_scan:
        bm = BM25([f"{r['text']} {r['description']}" for r in candidates])
        query_parts = [profile.system_type, profile.purpose,
                       " ".join(profile.network_interfaces), profile.architecture_notes]
        top = bm.top_n(" ".join(p for p in query_parts if p), n=top_n)
        keep_idx = {i for i, _ in top}
        candidates = [r for i, r in enumerate(candidates) if i in keep_idx]
        if progress:
            progress(f"BM25 shortlist: {len(candidates)} УБИ")

    done_numbers = done_numbers or set()
    todo = [r for r in candidates if r["Number"] not in done_numbers]
    results: List[Dict[str, Any]] = []

    # исключённые правилами фиксируем сразу (без LLM)
    for r in excluded:
        if r["Number"] in done_numbers:
            continue
        item = {
            "ubi_code": f"УБИ.{int(r['Number']):03d}" if r["Number"].isdigit() else r["Number"],
            "Number": r["Number"], "text": r["text"], "description": r["description"],
            "matches": False, "confidence": "высокая",
            "explanation": r["rule_reasons"] or "Исключена правилами по профилю объекта",
            "rule_score": r["rule_score"], "rule_reasons": r["rule_reasons"],
            "source": "rules",
        }
        results.append(item)
        if on_result:
            on_result(item)

    ctx = profile_context(profile)
    total_batches = (len(todo) + batch_size - 1) // batch_size if todo else 0

    def _flush_batch(batch: List[Dict[str, str]], parsed_map: Dict[str, Any],
                     only_parsed: bool) -> None:
        """Фиксирует результаты батча (append/on_result).

        При ``only_parsed=True`` (выход по отмене) сохраняются только
        фактически полученные от LLM результаты — в т.ч. успешно дообработанные
        одиночные; необработанные УБИ остаются для возобновления (без ложных
        записей llm_error).
        """
        for r in batch:
            num = r["Number"]
            item = parsed_map.get(_norm_ubi_number(num))
            if item:
                res = {
                    "ubi_code": f"УБИ.{int(num):03d}" if num.isdigit() else num,
                    "Number": num, "text": r["text"], "description": r["description"],
                    "matches": bool(item.get("matches", False)),
                    "confidence": _norm_conf(item.get("confidence")),
                    "explanation": str(item.get("explanation") or "").strip(),
                    "rule_score": r["rule_score"], "rule_reasons": r["rule_reasons"],
                    "source": "llm",
                }
            elif only_parsed:
                continue
            else:
                res = {
                    "ubi_code": f"УБИ.{int(num):03d}" if num.isdigit() else num,
                    "Number": num, "text": r["text"], "description": r["description"],
                    "matches": False,
                    "confidence": "низкая",
                    "explanation": (f"Не удалось получить корректный ответ LLM после "
                                    f"{retries * 2} попыток (ответ не получен, обрезан "
                                    f"или не парсится). Оцените применимость вручную."),
                    "rule_score": r["rule_score"], "rule_reasons": r["rule_reasons"],
                    "source": "llm_error",
                }
            results.append(res)
            if on_result:
                on_result(res)

    for bi, batch in enumerate(_chunk(todo, batch_size), start=1):
        if cancelled and cancelled():
            if progress:
                progress("Остановка по запросу пользователя (прогресс сохранён).")
            break
        user_prompt = UBI_USER_TEMPLATE.format(
            profile_summary=ctx, ubi_batch_text=_ubi_batch_text(batch)
        )
        cache_key = ""
        if cache is not None:
            cache_key = cache.key(PROMPT_VERSION, "ubi", provider.cfg.model,
                                  profile.fingerprint(),
                                  ",".join(r["Number"] for r in batch))
        parsed_map, _failed = call_llm_json(
            provider, UBI_SYSTEM_PROMPT, user_prompt,
            [r["Number"] for r in batch],
            key_normalizer=_norm_ubi_number,
            cache=cache, cache_key=cache_key, retries=retries,
            cancelled=cancelled, progress=progress,
        )
        stop = bool(cancelled and cancelled())
        if not stop:
            # поштучная дообработка УБИ, не полученных в батче
            for r in batch:
                if _norm_ubi_number(r["Number"]) in parsed_map:
                    continue
                if cancelled and cancelled():
                    stop = True
                    break
                if progress:
                    progress(f"УБИ {r['Number']}: нет ответа в батче — одиночный запрос.")
                single_prompt = UBI_USER_TEMPLATE.format(
                    profile_summary=ctx, ubi_batch_text=_ubi_batch_text([r])
                )
                single_map, _ = call_llm_json(
                    provider, UBI_SYSTEM_PROMPT, single_prompt, [r["Number"]],
                    key_normalizer=_norm_ubi_number, retries=retries,
                    cancelled=cancelled, progress=progress,
                )
                parsed_map.update(single_map)
            if cancelled and cancelled():
                stop = True

        # фиксируем всё, что успели получить в этом батче, ДО выхода по отмене
        _flush_batch(batch, parsed_map, only_parsed=stop)
        if stop:
            if progress:
                progress("Остановка по запросу пользователя (прогресс сохранён).")
            break

        if progress:
            progress(f"УБИ: батч {bi}/{total_batches} обработан")

    if progress:
        llm_done = [r for r in results if r["source"] in ("llm", "llm_error")]
        failed_cnt = len([r for r in llm_done if r["source"] == "llm_error"])
        progress(f"Без ответа LLM: {failed_cnt} из {len(llm_done)}")

    return results


# ======================================================================
# Этап: сопоставление актуальных УБИ с тактиками (раздел 7.2 / сценарии)
# ======================================================================
TACTICS_SYSTEM_PROMPT = (
    "Ты — специалист по информационной безопасности. Определяешь, какие способы (техники) "
    "реализации угроз применимы для конкретной угрозы (УБИ) с учётом профиля объекта. "
    "Отвечаешь строго валидным JSON-массивом без пояснений вне JSON."
)

TACTICS_USER_TEMPLATE = """Профиль объекта:
{profile_summary}

Угроза (УБИ {threat_num}): {threat_text}

Кандидаты — техники реализации:
{tactics_text}

Для КАЖДОЙ техники определи, может ли она использоваться нарушителем для реализации
именно этой угрозы на данном объекте. Если в профиле явно отсутствует нужная
технология/условие — matches=false.

Ответ — JSON-массив объектов вида:
[{{"tactic_id": "<код техники ТОЧНО как в списке, например Т3.1>", "matches": true/false, "confidence": "высокая|средняя|низкая", "explanation": "до 180 символов на русском"}}]
Верни по одному объекту на КАЖДУЮ технику из списка, без пояснений вне JSON.
"""


def analyze_tactics_for_threats(
    profile: Profile,
    provider: BaseProvider,
    actual_ubi: List[Dict[str, Any]],
    *,
    top_k: int = 12,
    batch_size: int = 6,
    done_pairs: Optional[Set[str]] = None,   # {"<ubi>|<tactic>"}
    cache=None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress: Optional[ProgressCb] = None,
    cancelled: Optional[CancelCb] = None,
    retries: int = LLM_RETRIES,
) -> List[Dict[str, Any]]:
    """Для каждой актуальной УБИ shortlist тактик по BM25 → LLM-оценка. Возвращает совпадения."""
    tactics = dicts.load_tactics()
    bm = BM25([f"{t['Category']} {t['Description']}" for t in tactics])
    done_pairs = done_pairs or set()
    ctx = profile_context(profile)
    results: List[Dict[str, Any]] = []

    def _tactics_prompt(ubi: Dict[str, Any], threat_text: str,
                        batch: List[Dict[str, str]]) -> str:
        tactics_text = "\n".join(
            f"{t['number']} [{t['Category']}] {t['Description'][:400]}" for t in batch
        )
        return TACTICS_USER_TEMPLATE.format(
            profile_summary=ctx,
            threat_num=ubi["Number"],
            threat_text=threat_text[:900],
            tactics_text=tactics_text,
        )

    for ti, ubi in enumerate(actual_ubi, start=1):
        if cancelled and cancelled():
            if progress:
                progress("Остановка по запросу пользователя (прогресс сохранён).")
            break
        threat_text = f"{ubi['text']}. {ubi.get('description', '')[:600]}"
        top = bm.top_n(threat_text, n=top_k)
        cand = [tactics[i] for i, score in top if score > 0]
        cand = [t for t in cand if f"{ubi['Number']}|{t['number']}" not in done_pairs]
        if not cand:
            if progress:
                progress(f"УБИ {ubi['Number']}: все тактики уже обработаны или нет кандидатов")
            continue

        stop = False
        for batch in _chunk(cand, batch_size):
            if cancelled and cancelled():
                stop = True
                break
            user_prompt = _tactics_prompt(ubi, threat_text, batch)
            cache_key = ""
            if cache is not None:
                cache_key = cache.key(PROMPT_VERSION, "tactics", provider.cfg.model,
                                      profile.fingerprint(), ubi["Number"],
                                      ",".join(t["number"] for t in batch))
            parsed_map, _failed = call_llm_json(
                provider, TACTICS_SYSTEM_PROMPT, user_prompt,
                [t["number"] for t in batch],
                key_normalizer=_norm_tactic_id,
                cache=cache, cache_key=cache_key, retries=retries,
                cancelled=cancelled, progress=progress,
            )
            if not (cancelled and cancelled()):
                # поштучная дообработка техник, не полученных в батче
                for t in batch:
                    if _norm_tactic_id(t["number"]) in parsed_map:
                        continue
                    if cancelled and cancelled():
                        stop = True
                        break
                    if progress:
                        progress(f"УБИ {ubi['Number']}, {t['number']}: нет ответа в батче — "
                                 f"одиночный запрос.")
                    single_map, _ = call_llm_json(
                        provider, TACTICS_SYSTEM_PROMPT,
                        _tactics_prompt(ubi, threat_text, [t]), [t["number"]],
                        key_normalizer=_norm_tactic_id, retries=retries,
                        cancelled=cancelled, progress=progress,
                    )
                    parsed_map.update(single_map)
            if cancelled and cancelled():
                stop = True

            # фиксируем всё, что успели получить в этом батче, ДО выхода по
            # отмене (включая успешно дообработанные одиночные); при отмене
            # необработанные техники остаются для возобновления (без llm_error)
            for t in batch:
                item = parsed_map.get(_norm_tactic_id(t["number"]))
                if item:
                    res = {
                        "threat_number": ubi["Number"],
                        "threat_code": ubi.get("ubi_code", f"УБИ.{ubi['Number']}"),
                        "threat_text": ubi["text"],
                        "tactic_id": t["number"],
                        "tactic_module": t["Module"],
                        "tactic_category": t["Category"],
                        "tactic": t["Description"],
                        "matches": bool(item.get("matches", False)),
                        "confidence": _norm_conf(item.get("confidence")),
                        "explanation": str(item.get("explanation") or "").strip(),
                        "source": "llm",
                    }
                elif stop:
                    continue
                else:
                    res = {
                        "threat_number": ubi["Number"],
                        "threat_code": ubi.get("ubi_code", f"УБИ.{ubi['Number']}"),
                        "threat_text": ubi["text"],
                        "tactic_id": t["number"],
                        "tactic_module": t["Module"],
                        "tactic_category": t["Category"],
                        "tactic": t["Description"],
                        "matches": False,
                        "confidence": "низкая",
                        "explanation": (f"Не удалось получить корректный ответ LLM после "
                                        f"{retries * 2} попыток (ответ не получен, обрезан "
                                        f"или не парсится). Оцените применимость вручную."),
                        "source": "llm_error",
                    }
                results.append(res)
                if on_result:
                    on_result(res)
            if stop:
                break

        if stop:
            if progress:
                progress("Остановка по запросу пользователя (прогресс сохранён).")
            break
        if progress:
            progress(f"Тактики: УБИ {ubi['Number']} ({ti}/{len(actual_ubi)}) обработана")

    return results


# ======================================================================
# Короткие текстовые генерации для разделов документа
# ======================================================================
TEXT_SYSTEM_PROMPT = (
    "Ты — специалист по информационной безопасности, готовишь текст раздела официального "
    "документа «Модель угроз безопасности информации» (по Методике ФСТЭК России от 05.02.2021). "
    "Пиши строгим канцелярским стилем, по-русски, без markdown, без списков вопросов, "
    "без упоминания того, что ты ИИ. Не выдумывай факты, отсутствующие в исходных данных. "
    "Опирайся на терминологию и структуру Методики ФСТЭК из приложенных выдержек, если они есть."
)


def generate_section_text(
    provider: BaseProvider,
    task: str,
    facts: str,
    *,
    max_chars: int = 4000,
    paragraphs: str = "2–5 абзаца",
    cache=None,
    cache_extra: str = "",
) -> str:
    """Генерация связного текста раздела документа по фактам профиля."""
    user = (
        f"Задача: {task}\n\nИсходные данные:\n{facts}\n\n"
        f"Объём: {paragraphs}, не более {max_chars} символов. Только текст раздела."
    )
    if cache is not None:
        key = cache.key(PROMPT_VERSION, "text", provider.cfg.model, task, facts, cache_extra)
        cached = cache.get(key)
        if cached:
            return cached
    out = (provider.generate(TEXT_SYSTEM_PROMPT, user, json_mode=False) or "").strip()
    if cache is not None and out:
        cache.put(key, out)
    return out


# ======================================================================
# Сценарий реализации конкретной угрозы (раздел 7.2)
# ======================================================================
def generate_scenario_text(
    provider: BaseProvider,
    profile: Profile,
    ubi_item: Dict[str, Any],
    matched_tactics: List[Dict[str, Any]],
    *,
    cache=None,
    extra_facts: str = "",
) -> str:
    """Связный сценарий реализации конкретной угрозы (2–4 абзаца):
    нарушитель (уровень возможностей) → вектор/техники из ``matched_tactics`` →
    объекты воздействия → возможные негативные последствия из профиля.

    ``extra_facts`` — дополнительный блок фактов (например, выдержки из
    Методики ФСТЭК), добавляемый в конец исходных данных промпта.

    При любой ошибке LLM возвращает "" (ошибки глушит вызывающий код pipeline).
    """
    ubi_code = ubi_item.get("ubi_code") or f"УБИ.{ubi_item.get('Number', '')}"
    tactics_text = "\n".join(
        f"- {t.get('tactic_id', '')} [{t.get('tactic_category', '')}] "
        f"{str(t.get('tactic', ''))[:300]}"
        for t in matched_tactics
    ) or "—"
    facts = (
        f"Угроза: {ubi_code}. {ubi_item.get('text', '')}\n"
        f"Описание угрозы: {str(ubi_item.get('description', ''))[:600]}\n\n"
        f"Применимые техники реализации:\n{tactics_text}\n\n"
        f"Профиль объекта:\n{profile_context(profile)}"
    )
    if extra_facts.strip():
        facts += "\n\n" + extra_facts.strip()
    task = (
        "Составь связный сценарий реализации указанной угрозы для раздела 7.2 "
        "модели угроз: какой нарушитель (с каким уровнем возможностей) может "
        "реализовать угрозу, через какие векторы и техники из списка, какие "
        "компоненты системы станут объектами воздействия и к каким негативным "
        "последствиям из профиля это может привести."
    )
    try:
        return generate_section_text(
            provider, task, facts,
            paragraphs="2–4 абзаца",
            cache=cache, cache_extra=f"scenario:{ubi_item.get('Number', '')}",
        )
    except Exception as e:
        logger.warning("Сценарий для %s не сформирован: %s", ubi_code, e)
        return ""
