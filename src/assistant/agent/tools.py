"""Инструменты поиска и ERP для агента."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

_WAREHOUSES = ("Москва", "Санкт-Петербург", "Казань", "Новосибирск")
_QUESTION = {"type": "string", "description": "Вопрос для поиска по базе знаний"}
_PART_ID = {"type": "string", "pattern": "^P-[0-9]{4}[A-Z]?$", "description": "Подтверждённый номер детали"}
_WAREHOUSE = {"type": "string", "enum": list(_WAREHOUSES)}
_QUANTITY = {"type": "integer", "minimum": 1, "description": "Количество, указанное пользователем"}
_MODEL = {"type": "string", "pattern": "^(AR|BZ|GS)-[0-9]{3}$"}


def _schema(name: str, description: str, **properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema("search_knowledge", "Найти конкретную деталь, замену, узел или регламент.", question=_QUESTION),
    _schema("summarize_topic", "Ответить на обзорный вопрос по сообществам графа.", question=_QUESTION),
    _schema("check_stock", "Проверить остаток детали на конкретном складе.", part_id=_PART_ID, warehouse=_WAREHOUSE),
    _schema("get_price", "Узнать цену детали в рублях.", part_id=_PART_ID),
    _schema(
        "create_service_order",
        "Подготовить сервисный заказ по явному запросу пользователя.",
        part_id=_PART_ID,
        warehouse=_WAREHOUSE,
        qty=_QUANTITY,
        model=_MODEL,
    ),
]
_PARAMETERS = {item["function"]["name"]: item["function"]["parameters"]["properties"] for item in TOOL_SCHEMAS}


def _is_code(value: str, kind: str) -> bool:
    return normalize_code(value) == value and code_type(value) == kind


def validate_arguments(name: str, arguments: dict) -> str | None:
    """Проверяет набор и значения аргументов до выполнения инструмента."""
    parameters = _PARAMETERS.get(name)
    if parameters is None:
        return f"Неизвестный инструмент: {name}."
    if arguments.keys() != parameters.keys():
        return f"Ожидаются параметры: {', '.join(parameters)}."
    for key, value in arguments.items():
        if key == "qty":
            if type(value) is not int or value <= 0:
                return "Количество должно быть положительным целым числом."
        elif not isinstance(value, str) or not value.strip():
            return f"Параметр {key} должен быть непустой строкой."
        elif key == "warehouse" and value not in _WAREHOUSES:
            return f"Неизвестный склад: {value}."
        elif key == "part_id" and not _is_code(value, "Part"):
            return "Укажите номер детали вида P-1042 или P-1042B."
        elif key == "model" and not _is_code(value, "Model"):
            return "Укажите код модели вида AR-210."
    return None


def _validated(name: str, function: Callable[..., dict]) -> Callable[..., dict]:
    def execute(**arguments) -> dict:
        if error := validate_arguments(name, arguments):
            return {"error": error}
        try:
            return function(**arguments)
        except ErpNotFound as exc:
            return {"error": f"Данных в ERP нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    return execute


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Связывает инструменты агента с поиском и клиентом ERP."""

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    functions = {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": erp.get_stock,
        "get_price": erp.get_price,
        "create_service_order": erp.create_order,
    }
    return {name: _validated(name, function) for name, function in functions.items()}
