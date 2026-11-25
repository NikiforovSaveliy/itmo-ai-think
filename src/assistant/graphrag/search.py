"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import combinations, pairwise

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from assistant.infra.evaluation import fabricated_facts
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, Community, Entity, KnowledgeBase, Relation

_logger = logging.getLogger(__name__)
_SINGLE_ANCHOR_DEPTH = 2
_INTER_ANCHOR_MAX_PATH = 3
_MAX_FACT_CHARS = 8000
_MIN_PARTIAL_SCORE = 1
_LOCAL_MAX_TOKENS = 1024
_MAP_MAX_TOKENS = 1024
_REDUCE_MAX_TOKENS = 1024

_NO_FACTS = "В базе знаний не найдено."
_LOCAL_PROMPT = (
    "Ответь на вопрос по-русски, используя только приведенные факты. "
    "Если спрашивают, какую деталь использовать, следуй связям REPLACED_BY "
    "от старой детали к последней замене. "
    f"Если фактов недостаточно, ответь: «{_NO_FACTS}»"
)
_LOCAL_PART_PROMPT = (
    " HAS_PART связывает модель с деталью, установленной по руководству. "
    "Исходная деталь связи REPLACED_BY снята с производства и заменена конечной деталью. "
    "Если спрашивают, где деталь была установлена, перечисли модели из HAS_PART с этой деталью."
)
_MAP_PROMPT = (
    "Ответь на вопрос по одной сводке. Сохрани относящиеся к вопросу факты и числа. "
    "Верни JSON с полями answer и score. Оцени полезность от 0 до 10. "
    "Поставь 0, если в сводке нет полезных фактов."
)
_SCOPED_MAP_PROMPT = (
    "Ответь на вопрос по одной сводке и приложенным фактам. "
    "Сохрани относящиеся к вопросу факты и числа. "
    "Верни JSON с полями answer и score. Оцени полезность от 0 до 10. "
    "Поставь 0, если данных недостаточно."
)
_OUT_OF_SCOPE = "\n\nСводка вне выбранной линейки. Верни пустой ответ с оценкой 0."
_LINE_FACTS_HEADER = "\n\nПроверенные данные графа по выбранной линейке:\n"
_MAP_LINE_PROMPT = (
    "\nПри расхождении чисел используй данные графа. Различай отказы моделей и узлов, а также исходные детали и замены."
)
_REDUCE_PROMPT = (
    "Объедини частичные ответы в один ответ по-русски. "
    "Опирайся на частичные ответы и приложенную сводку, если она есть. "
    "Не учитывай один факт дважды. "
    f"Если данных недостаточно, ответь: «{_NO_FACTS}»"
)
_REDUCE_LINE_PROMPT = (
    " Для статистики отказов и замен используй прежде всего проверенные данные графа. "
    "Различай итоги по моделям и по узлам, исходные детали и их замены. "
    "Отвечай ровно на заданный вопрос, без посторонних кодов."
)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient, *, options: SearchOptions | None = None) -> Answer:
    """Отвечает на вопрос по связям графа вокруг опорных сущностей."""
    _logger.info("local: поиск по графу для вопроса %.120r", question)
    options = options if options is not None else SearchOptions()
    anchors = _extract_anchors(question, kb.vocabulary)
    matches = KnowledgeSearch(kb).find_by_anchors(anchors, options)
    if matches is None:
        _logger.info("local: граф не дал ответа")
        return Answer(_NO_FACTS, refused=True)

    facts, sources = _format_local_facts(matches, options.max_fact_chars)
    _logger.info("local: отобрано %d фактов для ответа", len(facts))
    if not facts:
        return Answer(_NO_FACTS, refused=True)

    text = _answer_from_facts(question, facts, matches.anchor, llm, options.local_max_tokens)
    if _is_refusal(text):
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=sources, facts=facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient, *, options: SearchOptions | None = None) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага."""
    _logger.info("global: подготовка сводок для вопроса %.120r", question)
    options = options if options is not None else SearchOptions()
    search = KnowledgeSearch(kb)
    line_names = _extract_line_names(question, kb.vocabulary)
    context = search.find_by_lines(line_names)
    line_facts = _format_line_facts(context.line_facts)
    partials: list[str] = []
    chunk_ids = set(context.line_facts.evidence.chunk_ids)
    known_codes = set(context.line_facts.evidence.codes)
    for number, summary in enumerate(context.summaries, start=1):
        _logger.info("global: map %d/%d, сообщество %s", number, len(context.summaries), summary.community.community_id)
        partial = _answer_from_summary(question, summary, context.has_scope, line_facts, llm, options)
        if partial is None:
            continue
        partials.append(partial)
        evidence = search.community_evidence(summary.community)
        chunk_ids.update(evidence.chunk_ids)
        known_codes.update(evidence.codes)

    if not partials:
        _logger.info("global: нет подходящих частичных ответов")
        return Answer(_NO_FACTS, refused=True)

    _logger.info("global: reduce, объединение %d частичных ответов", len(partials))
    text = _merge_answers(question, partials, context.summaries, line_facts, llm, options.reduce_max_tokens)
    if _is_refusal(text) or fabricated_facts(text, question, known_codes, tool_text=""):
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=_document_ids(chunk_ids))


@dataclass(frozen=True)
class SearchOptions:
    """Настройки обхода графа, отбора фактов и вызовов модели."""

    single_anchor_depth: int = _SINGLE_ANCHOR_DEPTH
    inter_anchor_max_path: int = _INTER_ANCHOR_MAX_PATH
    max_fact_chars: int = _MAX_FACT_CHARS
    min_partial_score: int = _MIN_PARTIAL_SCORE
    local_max_tokens: int = _LOCAL_MAX_TOKENS
    map_max_tokens: int = _MAP_MAX_TOKENS
    reduce_max_tokens: int = _REDUCE_MAX_TOKENS

    def __post_init__(self) -> None:
        for name in (
            "single_anchor_depth",
            "inter_anchor_max_path",
            "max_fact_chars",
            "local_max_tokens",
            "map_max_tokens",
            "reduce_max_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом")
        if type(self.min_partial_score) is not int or not 1 <= self.min_partial_score <= 10:
            raise ValueError("min_partial_score должен быть целым числом от 1 до 10")


@dataclass(frozen=True)
class SearchAnchors:
    """ID опорных узлов, распознанных по кодам и по словарным названиям."""

    code_nodes: tuple[str, ...] = ()
    named_nodes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphNode(Entity):
    """Сущность вместе с чанками, в которых она встречается."""

    chunk_ids: tuple[str, ...]

    @classmethod
    def from_graph(cls, graph: nx.DiGraph, node_id: str) -> GraphNode:
        attrs = graph.nodes[node_id]
        return cls(type=attrs["type"], name=attrs["name"], chunk_ids=tuple(attrs["sources"]))


@dataclass(frozen=True)
class GraphEdge(Relation):
    """Направленная связь с весом и подтверждающими её чанками."""

    subject: GraphNode
    object: GraphNode
    weight: int
    chunk_ids: tuple[str, ...]

    @classmethod
    def from_graph(cls, graph: nx.DiGraph, start: str, end: str) -> GraphEdge:
        attrs = graph.edges[start, end]
        return cls(
            subject=GraphNode.from_graph(graph, start),
            type=attrs["type"],
            object=GraphNode.from_graph(graph, end),
            weight=attrs["weight"],
            chunk_ids=tuple(attrs["sources"]),
        )


@dataclass
class _LocalMatches:
    relations: list[GraphEdge]
    anchor: GraphNode | None


@dataclass
class _Evidence:
    chunk_ids: set[str] = field(default_factory=set)
    codes: set[str] = field(default_factory=set)


@dataclass(frozen=True, order=True)
class _PartReplacement:
    model: str
    original_part: str
    replacement_part: str


@dataclass
class _LineFacts:
    failures_by_model: Counter[str] = field(default_factory=Counter)
    failures_by_component: Counter[str] = field(default_factory=Counter)
    replacements: set[_PartReplacement] = field(default_factory=set)
    evidence: _Evidence = field(default_factory=_Evidence)

    def record_failure(self, failure: GraphEdge) -> None:
        """Вес FAILED входит в обе группировки: по модели и по отказавшему узлу."""
        self.failures_by_model[failure.subject.name] += failure.weight
        self.failures_by_component[failure.object.name] += failure.weight
        self.evidence.chunk_ids.update(failure.chunk_ids)

    def record_replacement(self, installation: GraphEdge, replacement: GraphEdge) -> None:
        change = _PartReplacement(installation.subject.name, installation.object.name, replacement.object.name)
        self.replacements.add(change)
        self.evidence.codes.update((change.original_part, change.replacement_part))
        self.evidence.chunk_ids.update(installation.chunk_ids)
        self.evidence.chunk_ids.update(replacement.chunk_ids)


@dataclass
class _ScopedSummary:
    community: Community
    in_scope: bool


@dataclass
class _OverviewContext:
    line_facts: _LineFacts
    has_scope: bool
    summaries: list[_ScopedSummary]


class _PartialAnswer(BaseModel):
    answer: str
    score: int = Field(strict=True, ge=0, le=10)


class KnowledgeSearch:
    """Собирает контекст по опорным узлам и названиям линеек."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._graph = kb.graph
        self._communities = kb.communities

    def find_by_anchors(self, anchors: SearchAnchors, options: SearchOptions) -> _LocalMatches | None:
        """Возвращает факты и источники по переданным опорным узлам."""
        nodes = self._existing_anchors(anchors)
        if not nodes:
            return None

        edges = self._local_edges(nodes, options)
        if len(nodes) > 1:
            self._add_replacements(edges)

        if not edges:
            return None
        anchor = GraphNode.from_graph(self._graph, nodes[0]) if len(nodes) == 1 else None
        relations = [GraphEdge.from_graph(self._graph, start, end) for start, end in edges]
        return _LocalMatches(relations, anchor)

    def find_by_lines(self, line_names: list[str]) -> _OverviewContext:
        """Собирает факты по указанным линейкам и определяет область поиска по сводкам."""
        lines = {Entity("ProductLine", name).node_id for name in line_names}
        models = self._line_models(lines)
        scope = lines | models
        summaries = [
            _ScopedSummary(community, not scope or bool(scope.intersection(community.nodes)))
            for community in self._communities
        ]
        return _OverviewContext(self._line_facts(models), bool(scope), summaries)

    def community_evidence(self, community: Community) -> _Evidence:
        """Собирает чанки сообщества и коды его узлов и исходящих соседей."""
        members = set(community.nodes)
        neighbors = {end for _, end in self._graph.out_edges(community.nodes)}
        evidence = _Evidence()
        for node_id in members | neighbors:
            node = GraphNode.from_graph(self._graph, node_id)
            evidence.codes.update(find_codes(node.name))
            if node_id in members:
                evidence.chunk_ids.update(node.chunk_ids)
        return evidence

    def _existing_anchors(self, anchors: SearchAnchors) -> list[str]:
        graph = self._graph
        if any(node not in graph for node in anchors.code_nodes):
            return []
        return list(dict.fromkeys(node for node in anchors.code_nodes + anchors.named_nodes if node in graph))

    def _local_edges(self, anchors: list[str], options: SearchOptions) -> set[tuple[str, str]]:
        """Выбирает связи в окрестности или на путях между опорными сущностями."""
        graph = self._graph
        undirected = graph.to_undirected(as_view=True)
        if len(anchors) == 1:
            depth = options.single_anchor_depth
            distances = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=depth)
            return {
                (start, end)
                for start, end in graph.subgraph(distances).edges
                if min(distances[start], distances[end]) < depth
            }

        edges = set()
        for start, end in combinations(anchors, 2):
            for path in nx.all_simple_paths(undirected, start, end, cutoff=options.inter_anchor_max_path):
                for left, right in pairwise(path):
                    if graph.has_edge(left, right):
                        edges.add((left, right))
                    if graph.has_edge(right, left):
                        edges.add((right, left))
        return edges

    def _add_replacements(self, edges: set[tuple[str, str]]) -> None:
        """Дополняет найденные факты цепочками замен и поставщиками новых деталей."""
        nodes = {node for edge in edges for node in edge}
        pending = [node for node in nodes if GraphNode.from_graph(self._graph, node).type == "Part"]
        visited: set[str] = set()
        while pending:
            part = pending.pop()
            if part in visited:
                continue
            visited.add(part)
            for edge in self._outgoing(part, "REPLACED_BY"):
                replacement = edge.object.node_id
                edges.add((part, replacement))
                pending.append(replacement)
                for supplier in self._outgoing(replacement, "SUPPLIED_BY"):
                    edges.add((replacement, supplier.object.node_id))

    def _outgoing(self, node_id: str, kind: str) -> Iterator[GraphEdge]:
        for target, attrs in self._graph[node_id].items():
            if attrs["type"] == kind:
                yield GraphEdge.from_graph(self._graph, node_id, target)

    def _line_models(self, lines: set[str]) -> set[str]:
        """Находит модели, связанные с линейками рёбрами BELONGS_TO."""
        return {model for model, _, attrs in self._graph.in_edges(lines, data=True) if attrs["type"] == "BELONGS_TO"}

    def _line_facts(self, models: set[str]) -> _LineFacts:
        """Учитывает отказы каждой модели и прямые замены её штатных деталей."""
        facts = _LineFacts()
        for model_id in sorted(models):
            model = GraphNode.from_graph(self._graph, model_id)
            facts.failures_by_model[model.name] = 0
            facts.evidence.codes.add(model.name)
            for failure in self._outgoing(model_id, "FAILED"):
                facts.record_failure(failure)
            for installation in self._outgoing(model_id, "HAS_PART"):
                for replacement in self._outgoing(installation.object.node_id, "REPLACED_BY"):
                    facts.record_replacement(installation, replacement)
        return facts


def _extract_anchors(question: str, vocabulary: dict[str, list[str]]) -> SearchAnchors:
    code_nodes = tuple(Entity(code_type(code), code).node_id for code in find_codes(question))
    named_nodes = tuple(Entity(kind, name).node_id for kind, name in find_names(question, vocabulary))
    return SearchAnchors(code_nodes=code_nodes, named_nodes=named_nodes)


def _format_local_facts(matches: _LocalMatches, max_chars: int) -> tuple[list[str], list[str]]:
    """Отбирает полные строки по бюджету промпта и источники вошедших в него связей."""

    def fact_priority(edge: GraphEdge) -> tuple[bool, bool, str, str]:
        start, end = edge.subject.node_id, edge.object.node_id
        outside_anchor = matches.anchor is not None and matches.anchor.node_id not in (start, end)
        return outside_anchor, edge.type != "REPLACED_BY", start, end

    facts: list[str] = []
    chunk_ids: set[str] = set()
    for edge in sorted(matches.relations, key=fact_priority):
        fact = f"{edge.subject.name} -{edge.type}-> {edge.object.name}"
        line_chars = len(f"{len(facts) + 1}. {fact}") + bool(facts)
        if line_chars <= max_chars:
            facts.append(fact)
            chunk_ids.update(edge.chunk_ids)
            max_chars -= line_chars
    return facts, _document_ids(chunk_ids)


def _answer_from_facts(
    question: str, facts: list[str], anchor: GraphNode | None, llm: LLMClient, max_tokens: int
) -> str:
    system = _LOCAL_PROMPT
    if anchor is not None and anchor.type == "Part":
        system += _LOCAL_PART_PROMPT
    user = f"Вопрос: {question}\n\nФакты:\n{_numbered_lines(facts)}"
    return _ask(llm, system, user, max_tokens=max_tokens).strip()


def _is_refusal(text: str) -> bool:
    return not text or _NO_FACTS.rstrip(".").casefold() in text.casefold()


def _extract_line_names(question: str, vocabulary: dict[str, list[str]]) -> list[str]:
    product_lines = {"ProductLine": vocabulary.get("ProductLine", [])}
    return [name for _, name in find_names(question, product_lines)]


def _format_line_facts(facts: _LineFacts) -> str:
    lines = [f"Отказы модели {model}: {count}" for model, count in facts.failures_by_model.items()]
    lines.extend(
        f"Отказы узла «{component}»: {count}" for component, count in facts.failures_by_component.most_common()
    )
    for change in sorted(facts.replacements):
        lines.append(
            f"Модель {change.model}: исходная деталь {change.original_part} снята с производства, "
            f"замена {change.replacement_part}."
        )
    return "\n".join(lines)


def _answer_from_summary(
    question: str, summary: _ScopedSummary, has_scope: bool, line_facts: str, llm: LLMClient, options: SearchOptions
) -> str | None:
    system = _SCOPED_MAP_PROMPT if has_scope else _MAP_PROMPT
    user = f"Вопрос: {question}\n\nСводка:\n{summary.community.summary}"
    if not summary.in_scope:
        user += _OUT_OF_SCOPE
    elif line_facts:
        user += _LINE_FACTS_HEADER + line_facts + _MAP_LINE_PROMPT

    content = _ask(llm, system, user, max_tokens=options.map_max_tokens, json_schema=MAP_SCHEMA)
    try:
        partial = _PartialAnswer.model_validate_json(content)
    except ValidationError:
        return None
    text = partial.answer.strip()
    if not summary.in_scope or partial.score < options.min_partial_score or not text:
        return None
    return text


def _merge_answers(
    question: str,
    partials: list[str],
    summaries: list[_ScopedSummary],
    line_facts: str,
    llm: LLMClient,
    max_tokens: int,
) -> str:
    answers = _numbered_lines(partials)
    system = _REDUCE_PROMPT
    if line_facts:
        answers += _LINE_FACTS_HEADER + line_facts
        system += _REDUCE_LINE_PROMPT
    elif len(summaries) == 1:
        summary = summaries[0].community.summary
        answers += f"\n\nСводка, подтверждающая частичный ответ:\n{summary}"
    user = f"Вопрос: {question}\n\nЧастичные ответы:\n{answers}"
    return _ask(llm, system, user, max_tokens=max_tokens).strip()


def _document_ids(chunk_ids: Iterable[str]) -> list[str]:
    return sorted({chunk.partition("#")[0] for chunk in chunk_ids})


def _numbered_lines(lines: list[str]) -> str:
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


def _ask(llm: LLMClient, system: str, user: str, *, max_tokens: int, json_schema: dict | None = None) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return llm.chat(messages, json_schema=json_schema, max_tokens=max_tokens).content
