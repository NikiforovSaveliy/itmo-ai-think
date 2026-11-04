"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from itertools import combinations

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, Entity, KnowledgeBase

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
    """Дополняет найденные факты цепочками замен по направлению ребер"""
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


def _local_facts(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Формирует список фактов в пределах лимита и собирает их документы"""
    facts: list[str] = []
    sources: set[str] = set()
    chars = 0
    for start, end in sorted(edges, key=lambda edge: (graph.edges[edge]["type"] != "REPLACED_BY", edge)):
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

    facts, sources = _local_facts(graph, edges)
    if not facts:
        return Answer(_NO_FACTS, refused=True)

    numbered_facts = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    response = llm.chat(
        [
            {"role": "system", "content": _LOCAL_PROMPT},
            {"role": "user", "content": f"Вопрос: {question}\n\nФакты:\n{numbered_facts}"},
        ]
    )
    text = response.content.strip()
    if not text or _NO_FACTS.rstrip(".").casefold() in text.casefold():
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=sources, facts=facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага"""
    partials: list[str] = []
    sources: set[str] = set()
    for community in kb.communities:
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос по одной сводке. Сохрани относящиеся к вопросу факты и числа. "
                        "Верни JSON с полями answer и score. Оцени полезность от 0 до 10. "
                        "Поставь 0, если в сводке нет полезных фактов."
                    ),
                },
                {"role": "user", "content": f"Вопрос: {question}\n\nСводка:\n{community.summary}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            partial = _PartialAnswer.model_validate_json(response.content)
        except ValidationError:
            continue
        if partial.score == 0 or not partial.answer.strip():
            continue
        partials.append(partial.answer.strip())
        for node in community.nodes:
            sources.update(chunk.split("#", 1)[0] for chunk in kb.graph.nodes[node]["sources"])

    if not partials:
        return Answer(_NO_FACTS, refused=True)

    answers = "\n".join(f"{index}. {answer}" for index, answer in enumerate(partials, start=1))
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы в один ответ по-русски. "
                    "Используй только данные из частичных ответов и не учитывай один факт дважды. "
                    f"Если данных недостаточно, ответь: «{_NO_FACTS}»"
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\n\nЧастичные ответы:\n{answers}"},
        ]
    )
    text = response.content.strip()
    if not text or _NO_FACTS.rstrip(".").casefold() in text.casefold():
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=sorted(sources))
