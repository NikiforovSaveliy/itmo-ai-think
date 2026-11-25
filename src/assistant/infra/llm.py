"""Клиент языковой модели: OpenAI-совместимый API, кеш на диске, учёт вызовов и токенов.

Все обращения к модели в проекте идут через LLMClient. Генерация детерминирована:
температура 0 и фиксированный seed. Одинаковый запрос второй раз берётся из кеша,
но в статистике считается как обычный вызов, чтобы стоимость режимов можно было сравнивать.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx

SEED = 7


class LLMError(RuntimeError):
    """Сервис модели недоступен или ответил ошибкой."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict  # если модель вернула не JSON, здесь {"_raw": "<строка>"}


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_message(self) -> dict:
        """Сообщение ассистента для истории диалога."""
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in self.tool_calls
            ]
        return message


def tool_message(call: ToolCall, result: dict) -> dict:
    """Сообщение с результатом инструмента для истории диалога."""
    return {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        embed_model: str,
        api_key: str = "ollama",
        cache_dir: Path | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.embed_model = embed_model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.stats = LLMStats()
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=600
        )

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Один ответ модели на историю сообщений.

        tools – описания инструментов в формате function calling. Если модель решила вызвать
        инструмент, вызовы лежат в `tool_calls`. json_schema – схема, которой должен
        соответствовать текст ответа.
        """
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "seed": SEED,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        if json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": json_schema}}
        data = self._cached("chat/completions", body)
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        response = LLMResponse(
            content=message.get("content") or "",
            tool_calls=[_parse_call(raw, i) for i, raw in enumerate(message.get("tool_calls") or [])],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        self.stats.calls += 1
        self.stats.prompt_tokens += response.prompt_tokens
        self.stats.completion_tokens += response.completion_tokens
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Эмбеддинги текстов в том же порядке."""
        vectors: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self._read(self._key("embeddings", {"model": self.embed_model, "input": text}))
            if cached is None:
                missing.append(text)
            else:
                vectors[text] = cached["embedding"]
        if missing:
            data = self._post("embeddings", {"model": self.embed_model, "input": missing})
            for text, row in zip(missing, data["data"], strict=True):
                vectors[text] = row["embedding"]
                self._write(
                    self._key("embeddings", {"model": self.embed_model, "input": text}), {"embedding": row["embedding"]}
                )
        return [vectors[text] for text in texts]

    def _cached(self, path: str, body: dict) -> dict:
        key = self._key(path, body)
        data = self._read(key)
        if data is not None:
            self.stats.cache_hits += 1
            return data
        data = self._post(path, body)
        self._write(key, data)
        return data

    def _post(self, path: str, body: dict) -> dict:
        try:
            response = self._http.post(path, json=body)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"сервис модели недоступен ({exc}). Проверьте, что он запущен, и адрес LLM_BASE_URL"
            ) from exc
        if response.status_code != 200:
            raise LLMError(f"сервис модели ответил кодом {response.status_code}: {response.text[:300]}")
        return response.json()

    @staticmethod
    def _key(path: str, body: dict) -> str:
        raw = json.dumps({"path": path, "body": body}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _read(self, key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _write(self, key: str, data: dict) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _parse_call(raw: dict, index: int) -> ToolCall:
    function = raw.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"_raw": arguments}
    return ToolCall(raw.get("id") or f"call_{index}", function.get("name", ""), arguments)
