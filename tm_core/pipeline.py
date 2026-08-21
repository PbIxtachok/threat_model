# -*- coding: utf-8 -*-
"""Конвейер генерации модели угроз: этапы с чекпоинтами и возобновлением.

Каждый этап:
  * читает готовый результат из <job>/<stage>.json, если он есть (resume);
  * «тяжёлые» LLM-этапы дополнительно пишут partial-чекпоинты после каждого
    батча (<stage>.partial.jsonl) — при остановке возобновление идёт с батча;
  * по завершении пишет <stage>.json и обновляет state.json.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from . import analysis, crypto_logic, dictionaries as dicts, intruder_logic, methodology, storage
from .providers import BaseProvider, LLMConfig, create_provider
from .schema import JobState, Profile, STAGE_ORDER, STAGE_TITLES

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, str], None]  # (stage, message)


class GenerationJob:
    """Управляемое задание генерации: запуск/возобновление/остановка."""

    def __init__(self, job_id: str, llm_cfg: LLMConfig,
                 llm_cfg_text: Optional[LLMConfig] = None,
                 progress: Optional[ProgressCb] = None,
                 ubi_full_scan: bool = True,
                 ubi_batch_size: int = 4,
                 tactics_top_k: int = 12):
        self.job_id = job_id
        self.llm_cfg = llm_cfg
        # конфиг для текстовых секций; если не задан — тот же, что для анализа
        self.llm_cfg_text = llm_cfg_text or llm_cfg
        self.cancel_event = threading.Event()
        self._progress_cb = progress
        self.ubi_full_scan = ubi_full_scan
        self.ubi_batch_size = ubi_batch_size
        self.tactics_top_k = tactics_top_k

        self.state: JobState = storage.load_job_state(job_id)  # type: ignore
        if self.state is None:
            raise ValueError(f"Задание {job_id} не найдено")
        self.profile: Profile = storage.load_job_profile(job_id)  # type: ignore
        if self.profile is None:
            raise ValueError(f"Профиль задания {job_id} не найден")
        self.cache = storage.LLMCache()
        self.provider: Optional[BaseProvider] = None        # анализ (JSON-задачи)
        self.text_provider: Optional[BaseProvider] = None   # тексты разделов

    # ------------------------------------------------------------------
    def _progress(self, stage: str, msg: str) -> None:
        logger.info("[%s] %s: %s", self.job_id, stage, msg)
        if self._progress_cb:
            try:
                self._progress_cb(stage, msg)
            except Exception:
                pass

    def cancel(self) -> None:
        self.cancel_event.set()

    def _cancelled(self) -> bool:
        return self.cancel_event.is_set()

    # ------------------------------------------------------------------
    @staticmethod
    def _methodology_block(query: str) -> str:
        """Выдержки из Методики ФСТЭК для подмешивания в facts (RAG-lite).

        Пустая строка, если файл Методики отсутствует или ничего не нашлось.
        """
        excerpts = methodology.excerpts_for(query)
        if not excerpts:
            return ""
        return ("Выдержки из Методики оценки угроз ФСТЭК России (справочно, "
                "для терминологии и структуры; не цитировать дословно):\n"
                + excerpts)

    def _methodology_query(self, task: str) -> str:
        """Запрос к индексу Методики: текст задачи + ключевые слова профиля."""
        p = self.profile
        keywords = " ".join([
            task, p.system_type, p.purpose, " ".join(p.info_kinds),
            " ".join(p.network_interfaces), " ".join(p.damage_types),
        ])
        return keywords

    # ------------------------------------------------------------------
    def _llm_text(self, stage: str, task: str, facts: str, **kw) -> str:
        """Генерация LLM-текста раздела с защитой от ошибок.

        Используется ``text_provider`` (конфигурация «LLM для текстов»).
        Если выдержки из Методики найдены — добавляются блоком в facts.
        При любой ошибке LLM — progress-сообщение и "" (этап не падает).
        """
        block = self._methodology_block(self._methodology_query(task))
        if block:
            facts = facts + "\n\n" + block
        try:
            return analysis.generate_section_text(
                self.text_provider, task, facts, cache=self.cache, **kw
            )
        except Exception as e:
            self._progress(stage, f"LLM недоступна ({e}); текст раздела пропущен.")
            return ""

    # ------------------------------------------------------------------
    def run(self) -> JobState:
        """Выполняет все незавершённые этапы. Возвращает итоговое состояние."""
        st = self.state
        st.status = "running"
        st.error = ""
        storage.save_job_state(st)
        needs_llm = any(st.stages.get(s) != "done"
                        for s in STAGE_ORDER if s != "s9_docx")
        if needs_llm:
            self.provider = create_provider(self.llm_cfg)
            self.text_provider = create_provider(self.llm_cfg_text)

        stage_funcs = {
            "s1_general": self._s1_general,
            "s2_description": self._s2_description,
            "s3_consequences": self._s3_consequences,
            "s4_impact_objects": self._s4_impact_objects,
            "s5_intruders": self._s5_intruders,
            "s6_ways": self._s6_ways,
            "s7_ubi": self._s7_ubi,
            "s8_scenarios": self._s8_scenarios,
            "s9_docx": self._s9_docx,
        }
        try:
            for stage in STAGE_ORDER:
                if self._cancelled():
                    st.status = "paused"
                    storage.save_job_state(st)
                    self._progress(stage, "Генерация приостановлена. Прогресс сохранён.")
                    return st
                if st.stages.get(stage) == "done":
                    continue
                st.current_stage = stage
                st.stages[stage] = "running"
                storage.save_job_state(st)
                self._progress(stage, f"Этап начат: {STAGE_TITLES.get(stage, stage)}")

                result = stage_funcs[stage]()

                if self._cancelled() and result is None:
                    st.stages[stage] = "pending"
                    st.status = "paused"
                    storage.save_job_state(st)
                    self._progress(stage, "Этап прерван, прогресс батчей сохранён.")
                    return st

                storage.save_stage_result(self.job_id, stage, result or {})
                st.stages[stage] = "done"
                # инвалидация downstream-этапов: если этап выполнен повторно,
                # все последующие «done»-этапы переводим в «pending» и удаляем
                # их файлы результатов (partial-чекпоинты батчей сохраняются),
                # чтобы результаты и DOCX не остались устаревшими
                invalidated = []
                for later in STAGE_ORDER[STAGE_ORDER.index(stage) + 1:]:
                    if st.stages.get(later) == "done":
                        st.stages[later] = "pending"
                        storage.delete_stage_result(self.job_id, later)
                        invalidated.append(later)
                storage.save_job_state(st)
                self._progress(stage, "Этап завершён.")
                if invalidated:
                    self._progress(stage, "Изменились исходные данные — будут "
                                          "пересчитаны этапы: " + ", ".join(invalidated))

            st.status = "done"
            st.current_stage = ""
            storage.save_job_state(st)
            self._progress("s9_docx", f"Готово. Документ: {st.docx_path}")
            return st
        except Exception as e:
            logger.exception("Ошибка генерации")
            st.status = "error"
            st.error = f"{type(e).__name__}: {e}"
            if st.current_stage:
                st.stages[st.current_stage] = "error"
            storage.save_job_state(st)
            self._progress(st.current_stage or "-", f"Ошибка: {st.error}")
            return st

    # ==================================================================
    # Этапы
    # ==================================================================
    def _s1_general(self) -> Dict[str, Any]:
        """Раздел 1 «Общие положения» — шаблонный текст + LLM-вступление."""
        p = self.profile
        intro_text = self._llm_text(
            "s1_general",
            "Напиши вводный текст (1–2 абзаца) раздела «Общие положения» модели "
            "угроз: цель разработки документа применительно к КОНКРЕТНОЙ системе "
            "(её назначение, оператор, установленный класс/уровень защищённости).",
            analysis.profile_context(p),
            paragraphs="1–2 абзаца", max_chars=2000,
        )
        return {
            "object_name": p.object_name,
            "operator_name": p.operator_name,
            "responsible": p.responsible,
            "developer_org": p.developer_org,
            "system_type": p.system_type,
            "protection_level": p.protection_level,
            "intro_text": intro_text,
        }

    def _s2_description(self) -> Dict[str, Any]:
        """Раздел 2 «Описание системы»: факты профиля + LLM-тексты подразделов."""
        p = self.profile
        result: Dict[str, Any] = {
            "purpose": p.purpose,
            "business_processes": p.business_processes,
            "scale": p.scale,
            "components": [c.__dict__ for c in p.components],
            "interfaces": p.network_interfaces,
            "user_groups": p.user_groups,
            "cloud_model": p.cloud_model,
            "cloud_details": p.cloud_details,
            "security_tools": p.security_tools,
            "info_kinds": p.info_kinds,
            "pdn_categories": p.pdn_categories,
            "pdn_volume": p.pdn_volume,
        }
        stage = "s2_description"
        facts = analysis.profile_context(p)
        result["purpose_text"] = self._llm_text(
            stage,
            "Составь связный текст о назначении информационной системы, решаемых "
            "задачах и основных (бизнес-)процессах, которые она обеспечивает, — "
            "для подраздела 2.1 модели угроз.",
            facts,
        )
        result["info_text"] = self._llm_text(
            stage,
            "Составь связный текст об информации, обрабатываемой в системе, и её "
            "значимости для оператора (виды информации, категории и объёмы "
            "персональных данных) — для подраздела 2.2 модели угроз.",
            facts,
        )
        result["architecture_text"] = self._llm_text(
            stage,
            "Составь связное описание архитектуры и условий функционирования "
            "информационной системы для раздела 2.3 модели угроз. Учти состав "
            "компонентов, интерфейсы и каналы взаимодействия, модель размещения "
            "(облако/ЦОД), применяемые средства защиты информации и особенности, "
            "указанные в исходных данных.",
            facts, max_chars=4500,
        )
        result["users_text"] = self._llm_text(
            stage,
            "Составь связный текст о группах пользователей системы и их ролях "
            "(включая администраторов и внешних участников, если они указаны) — "
            "для подраздела 2.4 модели угроз.",
            facts,
        )
        return result

    def _s3_consequences(self) -> Dict[str, Any]:
        """Раздел 3 «Негативные последствия»: выбор пользователя + обоснование LLM."""
        p = self.profile
        dmg = dicts.damages()["damage_types"]
        selected = [d for d in dmg if d["id"] in p.damage_types]
        custom = [s.strip() for s in p.consequences_custom.splitlines() if s.strip()]
        rows = []
        for d in selected:
            chosen = [c for c in d["consequences"] if c in p.consequences]
            rows.append({"id": d["id"], "name": d["name"], "consequences": chosen})
        facts = (
            analysis.profile_context(p)
            + "\nСостав компонентов: "
            + "; ".join(f"{c.name} ({c.ctype})" for c in p.components)
            + f"\nМасштаб и размещение: {p.scale or '—'}"
            + f"\nПрименяемые СЗИ/СКЗИ: {'; '.join(p.security_tools) or '—'}"
            + f"\nОбъём обрабатываемых ПДн: {p.pdn_volume or '—'}"
            + "\n\nВыбранные виды ущерба: "
            + ", ".join(f"{r['id']} ({r['name']})" for r in rows)
            + "\nПоследствия: " + "; ".join(sum([r["consequences"] for r in rows], []) + custom)
        )
        rationale = self._llm_text(
            "s3_consequences",
            "Обоснуй (2-3 абзаца), почему для данной системы определены указанные "
            "виды риска (ущерба) и негативные последствия. Учитывай состав "
            "компонентов, масштаб системы, применяемые средства защиты и объёмы "
            "обрабатываемых персональных данных.",
            facts,
            paragraphs="2–3 абзаца",
        )
        return {"rows": rows, "custom": custom, "rationale": rationale}

    def _s4_impact_objects(self) -> Dict[str, Any]:
        """Раздел 4 «Объекты воздействия»: компоненты × виды воздействия + LLM-обоснование."""
        p = self.profile
        kinds = dicts.misc()["impact_kinds"]
        rows = []
        for c in p.components:
            low = (c.ctype + " " + c.purpose).lower()
            impacts = ["Несанкционированный доступ к компоненту",
                       "Нарушение функционирования (работоспособности) компонента"]
            if any(k in low for k in ("баз", "субд", "хранени", "арм", "сервер", "веб", "почт")):
                impacts.insert(0, "Утечка (нарушение конфиденциальности) защищаемой информации")
                impacts.append("Несанкционированная модификация (подмена) информации")
                impacts.append("Уничтожение информации")
            if any(k in low for k in ("канал", "сет", "межсетев", "крипто", "vpn")):
                impacts.append("Перехват информации, передаваемой по каналам связи")
            if any(k in low for k in ("веб", "сервер", "субд")):
                impacts.append("Отказ в обслуживании (нарушение доступности)")
            impacts = [i for i in dict.fromkeys(impacts) if i in kinds]
            rows.append({"component": c.name, "ctype": c.ctype,
                         "location": c.location or "—", "impacts": impacts})
        facts = analysis.profile_context(p) + "\n\nОбъекты воздействия и виды воздействия:\n" + "\n".join(
            f"- {r['component']} ({r['ctype']}): " + "; ".join(r["impacts"]) for r in rows
        )
        analysis_text = self._llm_text(
            "s4_impact_objects",
            "Обоснуй выбор объектов воздействия угроз и видов воздействия на них "
            "для каждого типа компонентов системы (серверы, СУБД, АРМ, сетевое "
            "оборудование и т.д.) — для раздела 4 модели угроз.",
            facts,
        )
        return {"rows": rows, "analysis_text": analysis_text}

    def _s5_intruders(self) -> Dict[str, Any]:
        """Раздел 5 «Источники угроз»: нарушители, уровни, сговор — по правилам Методики."""
        p = self.profile
        rows = intruder_logic.intruders_by_damage(p.intruder_ids, p.damage_types)
        lvl = intruder_logic.max_level(p.intruder_ids)
        max_level = intruder_logic.level_info(lvl)
        facts = analysis.profile_context(p) + "\n\nАктуальные нарушители:\n" + "\n".join(
            f"- вид {t['id']} «{t['name']}» (категория: {t['category']}, уровень {t['level']})"
            for t in rows
        ) + (f"\nМаксимальный уровень возможностей: {lvl} ({max_level.get('title', '').lower()})."
             f"\nПодключение к Интернет: {'да' if p.has_internet else 'нет'}; "
             f"удалённый доступ: {'да' if p.has_remote_access else 'нет'}; "
             f"подрядные организации: {'да' if p.has_contractors else 'нет'}.")
        intruders_text = self._llm_text(
            "s5_intruders",
            "Обоснуй актуальность выбранных видов нарушителей и максимального "
            "уровня их возможностей (со ссылкой на таблицу 6.1 и таблицу 8.1 "
            "Методики оценки угроз безопасности информации). Учти наличие "
            "подключения к Интернет, удалённого доступа и привлечения подрядных "
            "организаций — для раздела 5 модели угроз.",
            facts,
        )
        return {
            "intruders": rows,
            "max_level": max_level,
            "collusion": [
                {"a_name": pr["a"]["name"], "a_id": pr["a"]["id"],
                 "b_name": pr["b"]["name"], "b_id": pr["b"]["id"]}
                for pr in intruder_logic.collusion_pairs(p.intruder_ids)
            ],
            "excluded_reason": p.intruders_excluded_reason,
            "intruders_text": intruders_text,
        }

    def _s6_ways(self) -> Dict[str, Any]:
        """Раздел 6 «Способы реализации»: интерфейсы × категории тактик + LLM-текст."""
        p = self.profile
        tactics = dicts.load_tactics()
        modules = []
        seen = set()
        for t in tactics:
            if t["Module"] not in seen:
                seen.add(t["Module"])
                modules.append(f"{t['Module']}. {t['Category'].rstrip('.')}")
        facts = analysis.profile_context(p) + (
            "\n\nИнтерфейсы и каналы взаимодействия: "
            + (", ".join(p.network_interfaces) or "—")
            + "\nКатегории способов реализации угроз:\n"
            + "\n".join(f"- {m}" for m in modules)
        )
        ways_text = self._llm_text(
            "s6_ways",
            "Составь связный текст о способах реализации угроз безопасности "
            "информации при имеющихся у системы интерфейсах и каналах "
            "взаимодействия и перечисленных категориях тактик (способов) "
            "реализации угроз — для раздела 6 модели угроз.",
            facts,
        )
        return {"interfaces": p.network_interfaces, "modules": modules,
                "ways_text": ways_text}

    def _s7_ubi(self) -> Optional[Dict[str, Any]]:
        """Раздел 7.1: применимость УБИ (LLM, батчи, partial-чекпоинты)."""
        stage = "s7_ubi"
        done_items = storage.load_partials(self.job_id, stage)
        # записи с неудачным ответом LLM НЕ считаются обработанными —
        # при возобновлении они переобрабатываются заново
        done_numbers = {str(i.get("Number")) for i in done_items
                        if i.get("source") not in ("llm_missing", "llm_error")}
        failed_before = len(done_items) - len([i for i in done_items
                                               if str(i.get("Number")) in done_numbers])
        if done_numbers:
            self._progress(stage, f"Возобновление: уже обработано {len(done_numbers)} УБИ")
        if failed_before:
            self._progress(stage, f"Переобработка {failed_before} записей с ошибкой LLM")

        def on_result(item: Dict[str, Any]) -> None:
            storage.append_partial(self.job_id, stage, item)

        analysis.analyze_ubi(
            self.profile, self.provider,
            batch_size=self.ubi_batch_size,
            full_scan=self.ubi_full_scan,
            done_numbers=done_numbers,
            cache=self.cache,
            on_result=on_result,
            progress=lambda m: self._progress(stage, m),
            cancelled=self._cancelled,
        )
        if self._cancelled():
            return None
        all_items = storage.load_partials(self.job_id, stage)
        # дедупликация по номеру (последняя запись выигрывает)
        by_num = {str(i.get("Number")): i for i in all_items}
        items = sorted(by_num.values(), key=lambda i: int(i["Number"]) if str(i["Number"]).isdigit() else 0)
        actual = [i for i in items if i.get("matches")]
        llm_failed = [i for i in items if i.get("source") == "llm_error"]
        stats = {"total": len(items), "actual": len(actual), "llm_failed": len(llm_failed)}
        self._progress(stage, f"Итог: {len(actual)} актуальных УБИ из {len(items)} рассмотренных")

        facts = analysis.profile_context(self.profile) + (
            f"\n\nВсего рассмотрено УБИ из БДУ ФСТЭК: {stats['total']} "
            f"(часть угроз предварительно исключена правилами по профилю объекта); "
            f"актуальными признаны: {stats['actual']}; без ответа LLM: {stats['llm_failed']}."
        )
        intro_text = self._llm_text(
            stage,
            "Напиши вводный текст (1–2 абзаца) к подразделу 7.1 «Перечень "
            "актуальных угроз»: как проводилась оценка возможности реализации "
            "угроз (анализ БДУ ФСТЭК с учётом профиля объекта, префильтр по "
            "правилам и оценка с применением LLM) и сколько угроз рассмотрено.",
            facts, paragraphs="1–2 абзаца", max_chars=2000,
        )
        actual_brief = "; ".join(
            f"{i.get('ubi_code', i.get('Number'))} «{str(i.get('text', ''))[:120]}»"
            for i in actual[:60]
        )
        summary_text = self._llm_text(
            stage,
            "Напиши обобщающий текст (2–3 абзаца) после таблицы актуальных "
            "угроз: какие группы угроз признаны актуальными, чем они обусловлены "
            "(архитектура, интерфейсы, обрабатываемая информация) и на что "
            "обратить внимание при выборе мер защиты.",
            facts + "\n\nАктуальные угрозы: " + (actual_brief or "—"),
            paragraphs="2–3 абзаца", cache_extra="s7_summary",
        )
        result: Dict[str, Any] = {"all": items, "actual": actual, "stats": stats,
                                  "intro_text": intro_text, "summary_text": summary_text}

        # --- приказ ФСБ № 378: тип актуальных угроз и требуемый класс СКЗИ ---
        skzi = crypto_logic.skzi_assessment(self.profile)
        if skzi is not None:
            skzi_facts = facts + (
                f"\n\nВ системе применяются средства криптографической защиты "
                f"информации. Уровень защищённости: {skzi['uz']}. "
                f"Актуальные типы угроз (по ПП РФ № 1119): "
                + ", ".join(str(t) for t in skzi["threat_types"]) + ". "
                "Требуемые классы СКЗИ (приказ ФСБ России № 378): "
                + "; ".join(f"тип {r['type']} → {r['class']}" for r in skzi["rows"])
                + f". Минимально требуемый класс СКЗИ: {skzi['min_class']}."
            )
            skzi_text = self._llm_text(
                stage,
                "Напиши обоснование (1–2 абзаца) требуемого класса СКЗИ по "
                "приказу ФСБ России от 10.07.2014 № 378: исходя из уровня "
                "защищённости ИСПДн, актуальных типов угроз (по постановлению "
                "Правительства РФ № 1119) и количества актуальных УБИ поясни, "
                "почему требуется СКЗИ именно такого класса.",
                skzi_facts, paragraphs="1–2 абзаца", max_chars=2000,
                cache_extra="s7_skzi",
            )
            result["skzi"] = skzi
            result["skzi_text"] = skzi_text
        return result

    def _s8_scenarios(self) -> Optional[Dict[str, Any]]:
        """Раздел 7.2: сценарии — сопоставление актуальных УБИ с тактиками."""
        stage = "s8_scenarios"
        s7 = storage.load_stage_result(self.job_id, "s7_ubi") or {}
        actual = s7.get("actual", [])
        if not actual:
            return {"pairs": [], "scenario_texts": {},
                    "note": "Актуальные УБИ не выявлены"}
        done_items = storage.load_partials(self.job_id, stage)
        # обработанными считаются только записи с ЯВНЫМ успешным источником:
        # записи с ошибкой LLM (llm_error) и записи СТАРОГО формата без поля
        # source (неудача выглядела как explanation == "Нет ответа LLM")
        # переобрабатываются при возобновлении
        done_pairs = {f"{i.get('threat_number')}|{i.get('tactic_id')}" for i in done_items
                      if i.get("source") in ("llm", "rules")}
        failed_before = len(done_items) - len([
            i for i in done_items
            if f"{i.get('threat_number')}|{i.get('tactic_id')}" in done_pairs
        ])
        if done_pairs:
            self._progress(stage, f"Возобновление: уже обработано {len(done_pairs)} пар УБИ-тактика")
        if failed_before:
            self._progress(stage, f"Переобработка {failed_before} пар с ошибкой LLM "
                                  f"или в старом формате (без поля source)")

        def on_result(item: Dict[str, Any]) -> None:
            storage.append_partial(self.job_id, stage, item)

        analysis.analyze_tactics_for_threats(
            self.profile, self.provider, actual,
            top_k=self.tactics_top_k,
            done_pairs=done_pairs,
            cache=self.cache,
            on_result=on_result,
            progress=lambda m: self._progress(stage, m),
            cancelled=self._cancelled,
        )
        if self._cancelled():
            return None
        all_items = storage.load_partials(self.job_id, stage)
        uniq = {f"{i.get('threat_number')}|{i.get('tactic_id')}": i for i in all_items}
        pairs = [i for i in uniq.values() if i.get("matches")]
        self._progress(stage, f"Итог: {len(pairs)} пар «УБИ — тактика»")

        # связные сценарии реализации для угроз с совпавшими тактиками
        scenario_texts: Dict[str, str] = {}
        by_threat: Dict[str, List[Dict[str, Any]]] = {}
        for pr in pairs:
            by_threat.setdefault(str(pr["threat_number"]), []).append(pr)
        ubi_by_num = {str(u.get("Number")): u for u in actual}
        for num, matched in by_threat.items():
            if self._cancelled():
                return None
            ubi_item = ubi_by_num.get(num)
            if not ubi_item:
                continue
            try:
                query = self._methodology_query(
                    f"Сценарий реализации угрозы: {ubi_item.get('text', '')} "
                    f"{str(ubi_item.get('description', ''))[:300]}"
                )
                txt = analysis.generate_scenario_text(
                    self.text_provider, self.profile, ubi_item, matched,
                    cache=self.cache, extra_facts=self._methodology_block(query),
                )
            except Exception as e:
                self._progress(stage, f"Сценарий для УБИ.{num} не сформирован ({e}).")
                txt = ""
            if txt:
                scenario_texts[num] = txt
        return {"pairs": pairs, "scenario_texts": scenario_texts}

    def _s9_docx(self) -> Dict[str, Any]:
        """Сборка итогового DOCX."""
        from . import docgen  # локальный импорт, чтобы python-docx не был нужен раньше времени
        sections = {s: storage.load_stage_result(self.job_id, s) or {} for s in STAGE_ORDER[:-1]}
        out_path = storage.job_dir(self.job_id) / "Модель угроз.docx"
        docgen.build_document(self.profile, sections, str(out_path))
        self.state.docx_path = str(out_path)
        storage.save_job_state(self.state)
        return {"docx_path": str(out_path)}


# ----------------------------------------------------------------------
def start_or_resume(job_id: str, llm_cfg_dict: Dict[str, Any],
                    llm_cfg_text_dict: Optional[Dict[str, Any]] = None,
                    progress: Optional[ProgressCb] = None,
                    **opts) -> GenerationJob:
    """Создаёт задание генерации.

    ``llm_cfg_dict`` — конфигурация LLM для анализа (JSON-задачи: УБИ,
    тактики); ``llm_cfg_text_dict`` — опциональная конфигурация LLM для
    текстов разделов и сценариев (если None — используется analysis-конфиг).
    """
    cfg = LLMConfig.from_dict(llm_cfg_dict)
    cfg_text = LLMConfig.from_dict(llm_cfg_text_dict) if llm_cfg_text_dict else None
    return GenerationJob(job_id, cfg, llm_cfg_text=cfg_text,
                         progress=progress, **opts)
