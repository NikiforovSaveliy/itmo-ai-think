"""Инструменты поиска и ERP для агента."""

from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass, replace
from functools import partial, wraps
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, ValidationInfo

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected, ErpUnavailable
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase


class Arguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


ToolFunction = Callable[..., dict]
ToolDecorator = Callable[[ToolFunction], ToolFunction]


def _handle_errors(function: ToolFunction) -> ToolFunction:
    @wraps(function)
    def wrapped(*args, **kwargs) -> dict:
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            return {"error": f"Ошибка выполнения инструмента: {exc}"}

    return wrapped


@dataclass(frozen=True)
class Tool:
    function: Callable[..., dict]
    input_schema: type[Arguments]
    response_schema: type[BaseModel]
    description: str
    handle_errors: ToolDecorator = _handle_errors

    def prepare(self, arguments: dict, *, context: dict | None = None) -> Callable[[], dict] | dict:
        """Проверяет аргументы до подтверждения заказа или обращения к ERP."""
        try:
            validated = self.input_schema.model_validate(arguments, context=context)
        except ValidationError as exc:
            return {"error": f"Неверные параметры инструмента: {exc}"}
        return partial(self._execute, validated)

    def __call__(self, **arguments) -> dict:
        prepared = self.prepare(arguments)
        return prepared() if callable(prepared) else prepared

    def _execute(self, arguments: Arguments) -> dict:
        result = self.function(**arguments.model_dump())
        if isinstance(result, dict) and "error" in result:
            return result
        try:
            self.response_schema.model_validate(result, strict=True)
        except ValidationError as exc:
            return {"error": f"Некорректный ответ инструмента: {exc}"}
        return result


_REGISTRY: dict[str, Tool] = {}


def register_tool(
    input_schema: type[Arguments], response_schema: type[BaseModel], *, handle_errors: ToolDecorator = _handle_errors
):
    """Регистрирует функцию, принимающую контекст первым аргументом, под её именем."""

    def decorate(function: Callable[..., dict]) -> Callable[..., dict]:
        wrapped = handle_errors(function)
        _REGISTRY[function.__name__] = Tool(
            wrapped, input_schema, response_schema, function.__doc__ or "", handle_errors
        )
        return wrapped

    return decorate


@dataclass(frozen=True)
class _Context:
    kb: KnowledgeBase
    llm: LLMClient
    erp: ErpClient


def get_tool_schemas(names: Container[str] | None = None) -> list[dict]:
    """Возвращает JSON-схемы зарегистрированных инструментов для LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description,
                "parameters": tool.input_schema.model_json_schema(),
            },
        }
        for name, tool in _REGISTRY.items()
        if names is None or name in names
    ]


def validate_arguments(name: str, arguments: dict) -> str | None:
    """Возвращает причину отказа, если инструмент или аргументы неизвестны."""
    if name not in _REGISTRY:
        return f"Неизвестный инструмент: {name}."
    prepared = _REGISTRY[name].prepare(arguments)
    return prepared["error"] if isinstance(prepared, dict) else None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Tool]:
    """Связывает инструменты агента с поиском и клиентом ERP."""
    context = _Context(kb, llm, erp)
    return {name: replace(tool, function=partial(tool.function, context)) for name, tool in _REGISTRY.items()}


def prepare_call(
    tools: Mapping[str, Callable[..., dict]], name: str, arguments: dict, known_codes: set[str]
) -> Callable[[], dict] | dict:
    """Готовит проверенный вызов либо возвращает сообщение об ошибке для модели."""
    if name not in _REGISTRY or name not in tools:
        return {"error": f"Неизвестный инструмент: {name}."}
    tool = tools[name]
    if not isinstance(tool, Tool):
        registered = _REGISTRY[name]
        tool = replace(registered, function=registered.handle_errors(tool))
    return tool.prepare(arguments, context={"known_codes": known_codes})


def _handle_erp_errors(function: ToolFunction) -> ToolFunction:
    @_handle_errors
    @wraps(function)
    def wrapped(*args, **kwargs) -> dict:
        for _ in range(3):
            try:
                return function(*args, **kwargs)
            except ErpUnavailable:
                continue
            except ErpNotFound as exc:
                return {"error": f"Данных в ERP нет: {exc}"}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}"}
        return {"error": "Данные ERP получить не удалось.", "unavailable": True}

    return wrapped


def _confirmed_part(part_id: str, info: ValidationInfo) -> str:
    known_codes = (info.context or {}).get("known_codes")
    if known_codes is not None and part_id not in known_codes:
        raise ValueError("Номер детали не подтверждён. Сначала вызови search_knowledge.")
    return part_id


Question = Annotated[str, Field(pattern=r"\S", description="Вопрос для поиска по базе знаний")]
PartId = Annotated[
    str,
    Field(pattern=r"^P-[0-9]{4}[A-Z]?$", description="Подтверждённый номер детали"),
    AfterValidator(_confirmed_part),
]
Warehouse = Literal["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


# search_knowledge


class SearchArgs(Arguments):
    question: Question


class SearchResult(BaseModel):
    answer: str
    sources: list[str]
    refused: bool
    facts: list[str]


@register_tool(SearchArgs, SearchResult)
def search_knowledge(context: _Context, question: str) -> dict:
    """Найти конкретную деталь, замену, узел или регламент."""
    answer = local_search(question, context.kb, context.llm)
    return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}


# summarize_topic


class SummaryArgs(Arguments):
    question: Question


class SummaryResult(BaseModel):
    answer: str
    sources: list[str]
    refused: bool


@register_tool(SummaryArgs, SummaryResult)
def summarize_topic(context: _Context, question: str) -> dict:
    """Ответить на обзорный вопрос по сообществам графа."""
    answer = global_search(question, context.kb, context.llm)
    return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}


# check_stock


class StockArgs(Arguments):
    part_id: PartId
    warehouse: Warehouse


class StockResult(BaseModel):
    part_id: str
    warehouse: str
    qty: int
    discontinued: bool


@register_tool(StockArgs, StockResult, handle_errors=_handle_erp_errors)
def check_stock(context: _Context, part_id: str, warehouse: str) -> dict:
    """Проверить остаток детали на конкретном складе."""
    return context.erp.get_stock(part_id=part_id, warehouse=warehouse)


# get_price


class PartArgs(Arguments):
    part_id: PartId


class PriceResult(BaseModel):
    part_id: str
    name: str
    price_rub: int
    discontinued: bool


@register_tool(PartArgs, PriceResult, handle_errors=_handle_erp_errors)
def get_price(context: _Context, part_id: str) -> dict:
    """Узнать цену детали в рублях."""
    return context.erp.get_price(part_id=part_id)


# create_service_order


class OrderArgs(Arguments):
    part_id: PartId
    warehouse: Warehouse
    qty: int = Field(gt=0, description="Количество, указанное пользователем")
    model: str = Field(pattern=r"^(AR|BZ|GS)-[0-9]{3}$")


class OrderResult(OrderArgs):
    order_id: str
    status: str


@register_tool(OrderArgs, OrderResult, handle_errors=_handle_erp_errors)
def create_service_order(context: _Context, part_id: str, warehouse: str, qty: int, model: str) -> dict:
    """Подготовить сервисный заказ по явному запросу пользователя."""
    return context.erp.create_order(part_id=part_id, warehouse=warehouse, qty=qty, model=model)


TOOL_SCHEMAS = get_tool_schemas()
