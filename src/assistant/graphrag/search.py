"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from collections import Counter
from itertools import combinations

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from assistant.infra.evaluation import fabricated_facts
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, Community, Entity, KnowledgeBase

_MAX_FACT_CHARS = 8000
_NO_FACTS = "В базе знаний не найдено."
_LOCAL_PROMPT = (
    "Ответь на вопрос по-русски, используя только приведенные факты. "
    "Если спрашивают, какую деталь использовать, следуй связям REPLACED_BY "
    "от старой детали к последней замене. "
    f"Если фактов недостаточно, ответь: «{_NO_FACTS}»"
)


class _PartialAnswer(BaseModel):
    answer: str
    score: int = Field(strict=True, ge=0, le=10)


def _local_edges(graph: nx.DiGraph, anchors: list[str]) -> set[tuple[str, str]]:
    """Выбирает связи в окрестности или на путях между опорными сущностями"""
    undirected = graph.to_undirected(as_view=True)
    if len(anchors) == 1:
        distances = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        return {
            (start, end) for start, end in graph.subgraph(distances).edges if min(distances[start], distances[end]) < 2
        }

    edges = set()
    for start, end in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
            for left, right in zip(path, path[1:], strict=False):
                if graph.has_edge(left, right):
                    edges.add((left, right))
                if graph.has_edge(right, left):
                    edges.add((right, left))
    return edges


def _add_replacements(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> None:
    """Дополняет найденные факты цепочками замен и поставщиками новых деталей"""
    pending = [node for node in {node for edge in edges for node in edge} if graph.nodes[node]["type"] == "Part"]
    visited: set[str] = set()
    while pending:
        part = pending.pop()
        if part in visited:
            continue
        visited.add(part)
        for _, replacement, attrs in graph.out_edges(part, data=True):
            if attrs["type"] == "REPLACED_BY":
                edges.add((part, replacement))
                pending.append(replacement)
                edges.update(
                    (replacement, supplier)
                    for _, supplier, link in graph.out_edges(replacement, data=True)
                    if link["type"] == "SUPPLIED_BY"
                )


def _local_facts(
    graph: nx.DiGraph, edges: set[tuple[str, str]], anchor: str | None = None
) -> tuple[list[str], list[str]]:
    """Формирует список фактов в пределах лимита и собирает их документы"""
    facts: list[str] = []
    sources: set[str] = set()
    chars = 0
    for start, end in sorted(
        edges,
        key=lambda edge: (anchor is not None and anchor not in edge, graph.edges[edge]["type"] != "REPLACED_BY", edge),
    ):
        attrs = graph.edges[start, end]
        fact = f"{graph.nodes[start]['name']} -{attrs['type']}-> {graph.nodes[end]['name']}"
        line_chars = len(f"{len(facts) + 1}. {fact}") + bool(facts)
        if chars + line_chars > _MAX_FACT_CHARS:
            continue
        facts.append(fact)
        chars += line_chars
        sources.update(chunk_id.split("#", 1)[0] for chunk_id in attrs["sources"])
    return facts, sorted(sources)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос по связям графа вокруг опорных сущностей"""
    graph = kb.graph
    code_anchors = [Entity(code_type(code), code).node_id for code in find_codes(question)]
    if any(node not in graph for node in code_anchors):
        return Answer(_NO_FACTS, refused=True)

    name_anchors = [Entity(kind, name).node_id for kind, name in find_names(question, kb.vocabulary)]
    anchors = list(dict.fromkeys(node for node in code_anchors + name_anchors if node in graph))
    if not anchors:
        return Answer(_NO_FACTS, refused=True)

    edges = _local_edges(graph, anchors)
    if len(anchors) > 1:
        _add_replacements(graph, edges)

    anchor = anchors[0] if len(anchors) == 1 else None
    facts, sources = _local_facts(graph, edges, anchor)
    if not facts:
        return Answer(_NO_FACTS, refused=True)

    prompt = _LOCAL_PROMPT
    if anchor is not None and graph.nodes[anchor]["type"] == "Part":
        prompt += (
            " HAS_PART связывает модель с деталью, установленной по руководству. "
            "Исходная деталь связи REPLACED_BY снята с производства и заменена конечной деталью. "
            "Если спрашивают, где деталь была установлена, перечисли модели из HAS_PART с этой деталью."
        )

    numbered_facts = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    response = llm.chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Вопрос: {question}\n\nФакты:\n{numbered_facts}"},
        ]
    )
    text = response.content.strip()
    if not text or _NO_FACTS.rstrip(".").casefold() in text.casefold():
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=sources, facts=facts)


def _product_lines(question: str, kb: KnowledgeBase) -> set[str]:
    """Находит идентификаторы названных линеек."""
    return {
        Entity(kind, name).node_id
        for kind, name in find_names(question, {"ProductLine": kb.vocabulary.get("ProductLine", [])})
    }


def _line_models(graph: nx.DiGraph, lines: set[str]) -> set[str]:
    """Находит модели, связанные с линейками рёбрами BELONGS_TO."""
    return {
        model
        for line in lines
        if line in graph
        for model, _, attrs in graph.in_edges(line, data=True)
        if attrs["type"] == "BELONGS_TO"
    }


def _line_facts(graph: nx.DiGraph, models: set[str]) -> tuple[list[str], set[str], set[str]]:
    """Собирает итоги отказов и связи замен по моделям выбранных линеек."""
    model_failures: Counter[str] = Counter()
    component_failures: Counter[str] = Counter()
    replacements: set[tuple[str, str, str]] = set()
    sources: set[str] = set()
    codes: set[str] = set()
    for model in sorted(models):
        model_name = graph.nodes[model]["name"]
        codes.add(model_name)
        model_failures[model_name] = 0
        for _, target, attrs in graph.out_edges(model, data=True):
            if attrs["type"] == "FAILED":
                model_failures[model_name] += attrs["weight"]
                component_failures[graph.nodes[target]["name"]] += attrs["weight"]
                sources.update(chunk.split("#", 1)[0] for chunk in attrs["sources"])
            elif attrs["type"] == "HAS_PART":
                for _, replacement, link in graph.out_edges(target, data=True):
                    if link["type"] != "REPLACED_BY":
                        continue
                    old_part = graph.nodes[target]["name"]
                    new_part = graph.nodes[replacement]["name"]
                    replacements.add((model_name, old_part, new_part))
                    codes.update((old_part, new_part))
                    sources.update(chunk.split("#", 1)[0] for chunk in attrs["sources"] + link["sources"])

    facts = [f"Отказы модели {model}: {count}" for model, count in model_failures.items()]
    facts.extend(f"Отказы узла «{component}»: {count}" for component, count in component_failures.most_common())
    facts.extend(
        f"Модель {model}: исходная деталь {old_part} снята с производства, замена {new_part}."
        for model, old_part, new_part in sorted(replacements)
    )
    return facts, sources, codes


def _community_codes(kb: KnowledgeBase, community: Community) -> set[str]:
    """Возвращает коды узлов и исходящих связей, использованных для сводки."""
    nodes = set(community.nodes)
    nodes.update(end for _, end in kb.graph.out_edges(community.nodes))
    return {code for node in nodes for code in find_codes(kb.graph.nodes[node]["name"])}


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага"""
    partials: list[str] = []
    sources: set[str] = set()
    known_codes: set[str] = set()
    lines = _product_lines(question, kb)
    models = _line_models(kb.graph, lines)
    scope = lines | models
    facts, graph_sources, graph_codes = _line_facts(kb.graph, models)
    graph_context = "\n".join(facts)
    for community in kb.communities:
        in_scope = not scope or bool(scope.intersection(community.nodes))
        context = f"Вопрос: {question}\n\nСводка:\n{community.summary}"
        if not in_scope:
            context += "\n\nСводка вне выбранной линейки. Верни пустой ответ с оценкой 0."
        elif facts:
            context += (
                "\n\nПроверенные данные графа по выбранной линейке:\n"
                f"{graph_context}\n"
                "При расхождении чисел используй данные графа. "
                "Различай отказы моделей и узлов, а также исходные детали и замены."
            )
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос по одной сводке и приложенным фактам. "
                        "Сохрани относящиеся к вопросу факты и числа. "
                        "Верни JSON с полями answer и score. Оцени полезность от 0 до 10. "
                        "Поставь 0, если данных недостаточно."
                    )
                    if scope
                    else (
                        "Ответь на вопрос по одной сводке. Сохрани относящиеся к вопросу факты и числа. "
                        "Верни JSON с полями answer и score. Оцени полезность от 0 до 10. "
                        "Поставь 0, если в сводке нет полезных фактов."
                    ),
                },
                {"role": "user", "content": context},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            partial = _PartialAnswer.model_validate_json(response.content)
        except ValidationError:
            continue
        if not in_scope or partial.score == 0 or not partial.answer.strip():
            continue
        partials.append(partial.answer.strip())
        known_codes.update(_community_codes(kb, community))
        for node in community.nodes:
            sources.update(chunk.split("#", 1)[0] for chunk in kb.graph.nodes[node]["sources"])

    if not partials:
        return Answer(_NO_FACTS, refused=True)

    answers = "\n".join(f"{index}. {answer}" for index, answer in enumerate(partials, start=1))
    if not facts and len(kb.communities) == 1:
        answers += f"\n\nСводка, подтверждающая частичный ответ:\n{kb.communities[0].summary}"
    system_prompt = (
        "Объедини частичные ответы в один ответ по-русски. "
        "Опирайся на частичные ответы и приложенную сводку, если она есть. "
        "Не учитывай один факт дважды. "
        f"Если данных недостаточно, ответь: «{_NO_FACTS}»"
    )
    if facts:
        answers += "\n\nПроверенные данные графа по выбранной линейке:\n" + graph_context
        system_prompt += (
            " Для статистики отказов и замен используй прежде всего проверенные данные графа. "
            "Различай итоги по моделям и по узлам, исходные детали и их замены. "
            "Отвечай ровно на заданный вопрос, без посторонних кодов."
        )
        known_codes.update(graph_codes)
        sources.update(graph_sources)
    response = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\n\nЧастичные ответы:\n{answers}"},
        ]
    )
    text = response.content.strip()
    if not text or _NO_FACTS.rstrip(".").casefold() in text.casefold():
        return Answer(_NO_FACTS, refused=True)
    if fabricated_facts(text, question, known_codes, tool_text=""):
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=sorted(sources))
