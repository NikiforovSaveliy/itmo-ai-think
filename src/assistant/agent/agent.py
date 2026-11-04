"""Цикл агента, проверки фактов и подтверждение заказа."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS, validate_arguments
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.evaluation import has_fact
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_NO_FACTS = "В базе знаний не найдено."
_ERP_UNAVAILABLE = "Данные ERP получить не удалось."
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


def _call_tool(function: Callable[..., dict], args: dict) -> dict:
    """Повторяет временно недоступный запрос ERP ещё два раза."""
    for attempt in range(3):
        try:
            return function(**args)
        except ErpUnavailable:
            if attempt == 2:
                return {"error": _ERP_UNAVAILABLE, "unavailable": True}
    raise AssertionError("Недостижимый исход запроса ERP")


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
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с инструментами и проверяет действия и итоговый ответ."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    results: list[dict] = []
    cache: dict[tuple[str, str], dict] = {}
    known_codes = set(find_codes(question))
    unavailable = False

    def finish(text: str, *, refused: bool = False, pending: dict | None = None) -> AgentResult:
        if unavailable and _ERP_UNAVAILABLE.casefold() not in text.casefold():
            text = f"{text} {_ERP_UNAVAILABLE}"
        return AgentResult(
            text,
            steps=steps,
            sources=sorted(sources),
            refused=refused,
            needs_confirmation=pending is not None,
            pending_action=pending,
        )

    while len(steps) < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            text = response.content.strip()
            if issues := _unsupported(text, known_codes, results):
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
            args = call.arguments
            error = validate_arguments(call.name, args)
            if call.name not in tools:
                error = f"Неизвестный инструмент: {call.name}."
            if not error and "part_id" in args and args["part_id"] not in known_codes:
                error = "Номер детали не подтверждён. Сначала вызови search_knowledge."
            if not error and call.name == _ORDER:
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

            if error:
                result = {"error": error}
            else:
                key = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if key not in cache:
                    cache[key] = _call_tool(tools[call.name], args)
                result = cache[key]

            steps.append(ToolStep(call.name, args, result=result, error=result.get("error")))
            messages.append(tool_message(call, result))
            if result.get("unavailable"):
                unavailable = True
            if "error" in result:
                continue

            results.append(result)
            sources.update(result.get("sources", []))
            if not result.get("refused"):
                known_codes.update(find_codes(json.dumps(result, ensure_ascii=False)))
            if call.name == _ORDER and result.get("order_id"):
                return finish(f"Заказ {result['order_id']} создан.")

    return finish(_partial_answer(results))
