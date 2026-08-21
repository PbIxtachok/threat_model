# -*- coding: utf-8 -*-
"""Абстракция LLM-провайдеров.

Поддерживаются:
  * ``openai``  — любой OpenAI-совместимый API (Ollama /v1, LM Studio, vLLM,
                  llama.cpp server, OpenRouter, YandexGPT-компат. и т.д.);
  * ``ollama``  — нативный API Ollama (/api/chat), умеет format=json;
  * ``anthropic`` — облачный Anthropic Claude (Messages API);
  * ``transformers`` — локальная модель in-process (опционально, тяжёлая зависимость);
  * ``mock``    — детерминированные ответы для тестов и отладки UI.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600  # локальная LLM может отвечать очень долго


@dataclass
class LLMConfig:
    provider: str = "ollama"            # ollama | openai | anthropic | transformers | mock
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:14b-instruct"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = DEFAULT_TIMEOUT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LLMConfig":
        known = {f for f in LLMConfig.__dataclass_fields__}
        return LLMConfig(**{k: v for k, v in (d or {}).items() if k in known})


class LLMError(RuntimeError):
    pass


class BaseProvider:
    name = "base"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        raise NotImplementedError

    def test_connection(self) -> str:
        """Возвращает строку статуса либо бросает LLMError."""
        out = self.generate(
            "Ты — эхо-сервис.", "Ответь одним словом: готов", json_mode=False
        )
        return f"Подключение успешно. Ответ модели: {out[:100]}"


class OpenAICompatibleProvider(BaseProvider):
    name = "openai"

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        url = self.cfg.base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.cfg.timeout)
        except requests.RequestException as e:
            raise LLMError(f"Ошибка сети при обращении к {url}: {e}") from e
        if r.status_code == 400 and json_mode:
            # некоторые серверы не поддерживают response_format — повторяем без него
            payload.pop("response_format", None)
            r = requests.post(url, json=payload, headers=headers, timeout=self.cfg.timeout)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Неожиданный ответ API: {json.dumps(data)[:300]}") from e


class OllamaProvider(BaseProvider):
    name = "ollama"

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(url, json=payload, timeout=self.cfg.timeout)
        except requests.RequestException as e:
            raise LLMError(f"Ошибка сети при обращении к {url}: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        return (data.get("message") or {}).get("content", "")


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    API_URL = "https://api.anthropic.com/v1/messages"

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.cfg.api_key:
            raise LLMError("Не задан API-ключ Anthropic.")
        headers = {
            "x-api-key": self.cfg.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if json_mode:
            user = user + "\n\nОтветь строго валидным JSON без пояснений вне JSON."
        payload = {
            "model": self.cfg.model or "claude-sonnet-4-5",
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        base = self.cfg.base_url.strip()
        url = (base.rstrip("/") + "/v1/messages") if base and "anthropic.com" not in base else self.API_URL
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.cfg.timeout)
        except requests.RequestException as e:
            raise LLMError(f"Ошибка сети при обращении к {url}: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        parts = data.get("content") or []
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


class TransformersProvider(BaseProvider):
    """Локальная модель in-process (как в исходной версии). Требует torch/transformers."""

    name = "transformers"
    _pipe = None  # кэш пайплайна на процесс

    def _ensure_pipe(self):
        if TransformersProvider._pipe is not None:
            return
        try:
            import torch  # noqa
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise LLMError(
                "Пакеты torch/transformers не установлены. Установите extras: pip install -r requirements-local.txt"
            ) from e
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        TransformersProvider._pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.cfg.max_tokens,
            do_sample=self.cfg.temperature > 0,
            temperature=max(self.cfg.temperature, 0.01),
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        self._ensure_pipe()
        pipe = TransformersProvider._pipe
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = pipe(prompt)
        return (out[0].get("generated_text") or "").strip() if out else ""


class MockProvider(BaseProvider):
    """Детерминированные правдоподобные ответы — для тестов и проверки UI без LLM."""

    name = "mock"

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        if not json_mode:
            return ("Тестовый текст (mock-провайдер). Описание сформировано без LLM "
                    "для проверки работоспособности конвейера.")
        # тактики: вернуть matches по кодам Тx.y (маркер — формат ответа с tactic_id)
        if "tactic_id" in user:
            tactics = re.findall(r"(Т\d+\.\d+)", user)
            items = [
                {
                    "tactic_id": t,
                    "matches": (i % 2 == 0),
                    "confidence": "средняя",
                    "explanation": f"Mock-оценка применимости {t}.",
                }
                for i, t in enumerate(dict.fromkeys(tactics))
            ]
            return json.dumps(items, ensure_ascii=False)
        # УБИ-батч: вернуть решение по каждому номеру из запроса
        numbers = re.findall(r"УБИ\s+(\d+)", user)
        if numbers:
            items = [
                {
                    "Number": n,
                    "matches": int(n) % 3 != 0,
                    "confidence": "средняя",
                    "explanation": f"Mock-оценка применимости УБИ {n} к объекту.",
                }
                for n in numbers
            ]
            return json.dumps(items, ensure_ascii=False)
        return json.dumps({"result": "ok", "explanation": "mock"}, ensure_ascii=False)


PROVIDERS = {
    "openai": OpenAICompatibleProvider,
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "transformers": TransformersProvider,
    "mock": MockProvider,
}

PROVIDER_LABELS = {
    "ollama": "Ollama (локально)",
    "openai": "OpenAI-совместимый API (LM Studio, vLLM, llama.cpp…)",
    "anthropic": "Anthropic Claude (облако)",
    "transformers": "Локальный transformers (in-process)",
    "mock": "Mock (тест без LLM)",
}


def create_provider(cfg: LLMConfig) -> BaseProvider:
    cls = PROVIDERS.get(cfg.provider)
    if cls is None:
        raise LLMError(f"Неизвестный провайдер: {cfg.provider}")
    return cls(cfg)


# ----------------------------------------------------------------------
def safe_parse_json(text: str) -> Any:
    """Максимально терпимый парсинг JSON из ответа LLM."""
    if not text:
        return None
    t = str(text).strip()
    t = re.sub(r"^```[\w-]*\s*|\s*```$", "", t, flags=re.MULTILINE).strip()
    candidates = [t]
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        c = c.replace("{{", "{").replace("}}", "}")
        c = _repair_string_newlines(c)
        attempts = [c]
        no_commas = _strip_trailing_commas(c)
        if no_commas != c:
            attempts.append(no_commas)
        for base in (no_commas, c):
            repaired = _repair_truncated_array(base)
            if repaired and repaired not in attempts:
                attempts.append(repaired)
        for a in attempts:
            try:
                return json.loads(a)
            except Exception:
                continue
    return None


def _strip_trailing_commas(text: str) -> str:
    """Удаляет «висячие» запятые перед закрывающими `]`/`}` (частая ошибка LLM).

    Посимвольная обработка с учётом строковых литералов (как в
    ``_repair_string_newlines``): запятые внутри строк (например, «, }» в
    текстовом значении) не затрагиваются.
    """
    out: list = []
    in_str, esc = False, False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue
        if ch == '"':
            out.append(ch)
            in_str = not in_str
            i += 1
            continue
        if in_str:
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            # вне строки: если после запятой (и пробельных символов) идёт
            # закрывающая скобка — запятую отбрасываем
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_truncated_array(text: str) -> Optional[str]:
    """Достраивает ОБРЕЗАННЫЙ JSON-массив: обрезает текст до последнего
    корректно закрытого объекта `}` внутри массива и добавляет `]`.

    Возвращает восстановленный текст либо None, если ремонт неприменим.
    """
    start = text.find("[")
    if start < 0:
        return None
    body = text[start:]
    if "]" in body:
        return None  # массив закрыт — это не случай обрезки
    # последняя `}` вне строковых литералов
    in_str, esc, last_close = False, False, -1
    for i, ch in enumerate(body):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "}":
            last_close = i
    if last_close < 0:
        return None
    return body[:last_close + 1] + "]"


def _repair_string_newlines(text: str) -> str:
    out, in_str, esc = [], False, False
    for ch in text:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = not in_str
            continue
        if in_str and ch in "\r\n":
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)
