"""Цикл агента, проверки фактов и подтверждение заказа."""

import json
import logging
import re
from collections.abc import Callable, Mapping
from time import monotonic

from assistant.agent.tools import get_tool_schemas, prepare_call
from assistant.infra.evaluation import has_fact
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_logger = logging.getLogger(__name__)
_NO_FACTS = "В базе знаний не найдено."
_ORDER = "create_service_order"
_WITH_UNIT = re.compile(r"(?<![\w-])(\d+(?:[ \u00a0\u202f]\d{3})*)\s*(руб\w*|₽|шт\w*|единиц\w*)", re.IGNORECASE)
_SYSTEM_PROMPT = (
    "Ты ассистент сервисной службы. Отвечай по-русски по результатам инструментов. "
    "Конкретные детали и замены ищи через search_knowledge, обзорные сведения через summarize_topic. "
    "Чтобы узнать цену или остаток, сначала найди подтверждённую деталь, затем вызови get_price или check_stock. "
    "Если есть замена, используй актуальную деталь. Не выдумывай коды, цены, остатки и номера заказов. "
    "Нулевой остаток, неизвестная деталь и недоступная ERP означают разные вещи. "
    "Вызывай create_service_order только по явной просьбе оформить заказ и с указанным количеством. "
    f"Если в базе нет ответа, скажи: «{_NO_FACTS}»"
)


def _unsupported(text: str, known_codes: set[str], results: list[dict]) -> list[str]:
    """Находит неподтверждённые коды, цены и количества в ответе."""
    prices = {row["price_rub"] for row in results if "price_rub" in row}
    quantities = {row["qty"] for row in results if "qty" in row and "order_id" not in row}
    issues = [code for code in find_codes(text) if code not in known_codes]
    for raw, unit in _WITH_UNIT.findall(text):
        number = int(re.sub(r"\s", "", raw))
        allowed = prices if unit.casefold().startswith("руб") or unit == "₽" else quantities
        if number not in allowed:
            issues.append(f"{raw} {unit}")
    return issues


def _partial_answer(results: list[dict]) -> str:
    """Перечисляет полученные сведения при исчерпании лимита."""
    lines = ["Достигнут лимит вызовов инструментов. Работа прервана."]
    for row in results:
        if row.get("answer") and not row.get("refused"):
            lines.append(row["answer"])
        if "price_rub" in row:
            lines.append(f"{row['part_id']}: цена {row['price_rub']} руб.")
        if "qty" in row and "warehouse" in row:
            lines.append(f"{row['part_id']}, склад {row['warehouse']}: остаток {row['qty']} шт.")
    return "\n".join(dict.fromkeys(lines))


def run_agent(
    question: str,
    llm: LLMClient,
    tools: Mapping[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с инструментами и проверяет действия и итоговый ответ."""
    _logger.info("agent: вопрос %.120r, лимит инструментов %d", question, max_steps)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    results: list[dict] = []
    cache: dict[tuple[str, str], dict] = {}
    known_codes = set(find_codes(question))
    unavailable_message = ""
    schemas = get_tool_schemas(tools)

    def finish(text: str, *, refused: bool = False, pending: dict | None = None) -> AgentResult:
        _logger.info(
            "agent: завершён после %d шагов, отказ=%s, подтверждение=%s", len(steps), refused, pending is not None
        )
        if unavailable_message and unavailable_message.casefold() not in text.casefold():
            text = f"{text} {unavailable_message}"
        return AgentResult(
            text,
            steps=steps,
            sources=sorted(sources),
            refused=refused,
            needs_confirmation=pending is not None,
            pending_action=pending,
        )

    while len(steps) < max_steps:
        _logger.info("agent: запрос к модели, выполнено шагов %d/%d", len(steps), max_steps)
        response = llm.chat(messages, tools=schemas)
        messages.append(response.as_message())
        if not response.tool_calls:
            text = response.content.strip()
            if issues := _unsupported(text, known_codes, results):
                _logger.info("agent: ответ отправлен модели на исправление, расхождений %d", len(issues))
                messages.append(
                    {
                        "role": "user",
                        "content": f"Исправь ответ без неподтверждённых фактов: {', '.join(issues)}.",
                    }
                )
                text = llm.chat(messages).content.strip()
            if not text or _unsupported(text, known_codes, results):
                return finish(_NO_FACTS, refused=True)
            if _NO_FACTS.rstrip(".").casefold() in text.casefold() or (
                results and all(result.get("refused") for result in results)
            ):
                return finish(_NO_FACTS, refused=True)
            return finish(text)

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                break
            _logger.info("agent: шаг %d/%d, инструмент %.80r", len(steps) + 1, max_steps, call.name)
            started = monotonic()
            args = call.arguments
            prepared = prepare_call(tools, call.name, args, known_codes)
            if callable(prepared) and call.name == _ORDER:
                if not has_fact(question, [str(args["qty"])]):
                    steps.append(ToolStep(call.name, args, error="Количество не указано пользователем."))
                    return finish("Сколько штук нужно заказать?")
                if not confirm:
                    steps.append(
                        ToolStep(
                            call.name,
                            args,
                            result={"status": "pending_confirmation", "qty": args["qty"]},
                            error="Заказ ожидает подтверждения.",
                        )
                    )
                    return finish(
                        f"Подтверди заказ: {args['qty']} шт. {args['part_id']} для {args['model']}, "
                        f"склад {args['warehouse']}.",
                        pending={"tool": call.name, "args": args},
                    )

            if isinstance(prepared, dict):
                result = prepared
            else:
                key = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if key in cache:
                    _logger.info("agent: повторный вызов %.80r, результат из кеша", call.name)
                else:
                    cache[key] = prepared()
                result = cache[key]

            _logger.info("agent: шаг завершён за %.2f с, ошибка=%s", monotonic() - started, "error" in result)
            steps.append(ToolStep(call.name, args, result=result, error=result.get("error")))
            messages.append(tool_message(call, result))
            if result.get("unavailable"):
                unavailable_message = result["error"]
            if "error" in result:
                continue

            results.append(result)
            sources.update(result.get("sources", []))
            if not result.get("refused"):
                known_codes.update(find_codes(json.dumps(result, ensure_ascii=False)))
            if call.name == _ORDER and result.get("order_id"):
                return finish(f"Заказ {result['order_id']} создан.")

    _logger.info("agent: достигнут лимит шагов (%d)", max_steps)
    return finish(_partial_answer(results))
