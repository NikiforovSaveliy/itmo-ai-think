"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from assistant.infra.evaluation import fabricated_facts
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, Community, Entity, KnowledgeBase

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


class KnowledgeSearch:
    """Собирает контекст по опорным узлам и названиям линеек."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._graph = kb.graph
        self._communities = kb.communities

    def find_by_anchors(self, anchors: SearchAnchors, options: SearchOptions) -> _LocalContext | None:
        """Возвращает факты и источники по переданным опорным узлам."""
        nodes = self._existing_anchors(anchors)
        if not nodes:
            return None

        edges = self._local_edges(nodes, options)
        if len(nodes) > 1:
            self._add_replacements(edges)

        anchor = nodes[0] if len(nodes) == 1 else None
        facts, sources = self._local_facts(edges, anchor, options.max_fact_chars)
        if not facts:
            return None

        single_part = anchor is not None and self._graph.nodes[anchor]["type"] == "Part"
        return _LocalContext(facts, sources, single_part)

    def find_by_lines(self, line_names: list[str]) -> _OverviewContext:
        """Собирает факты по указанным линейкам и определяет область поиска по сводкам."""
        lines = {Entity("ProductLine", name).node_id for name in line_names}
        models = self._line_models(lines)
        scope = lines | models
        facts, sources, codes = self._line_facts(models)
        summaries = [
            _ScopedSummary(community, not scope or bool(scope.intersection(community.nodes)))
            for community in self._communities
        ]
        return _OverviewContext("\n".join(facts), sources, codes, bool(scope), summaries)

    def community_evidence(self, community: Community) -> tuple[set[str], set[str]]:
        """Возвращает документы и коды, использованные для принятой сводки."""
        graph = self._graph
        nodes = set(community.nodes)
        nodes.update(end for _, end in graph.out_edges(community.nodes))
        codes = {code for node in nodes for code in find_codes(graph.nodes[node]["name"])}
        sources = {chunk.split("#", 1)[0] for node in community.nodes for chunk in graph.nodes[node]["sources"]}
        return sources, codes

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
                for left, right in zip(path, path[1:], strict=False):
                    if graph.has_edge(left, right):
                        edges.add((left, right))
                    if graph.has_edge(right, left):
                        edges.add((right, left))
        return edges

    def _add_replacements(self, edges: set[tuple[str, str]]) -> None:
        """Дополняет найденные факты цепочками замен и поставщиками новых деталей."""
        graph = self._graph
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
        self, edges: set[tuple[str, str]], anchor: str | None, max_chars: int
    ) -> tuple[list[str], list[str]]:
        """Формирует список фактов в пределах лимита и собирает их документы."""
        graph = self._graph

        def fact_priority(edge: tuple[str, str]) -> tuple[bool, bool, tuple[str, str]]:
            return anchor is not None and anchor not in edge, graph.edges[edge]["type"] != "REPLACED_BY", edge

        facts: list[str] = []
        sources: set[str] = set()
        chars = 0
        for start, end in sorted(edges, key=fact_priority):
            attrs = graph.edges[start, end]
            fact = f"{graph.nodes[start]['name']} -{attrs['type']}-> {graph.nodes[end]['name']}"
            line_chars = len(f"{len(facts) + 1}. {fact}") + bool(facts)
            if chars + line_chars > max_chars:
                continue
            facts.append(fact)
            chars += line_chars
            sources.update(chunk_id.split("#", 1)[0] for chunk_id in attrs["sources"])
        return facts, sorted(sources)

    def _line_models(self, lines: set[str]) -> set[str]:
        """Находит модели, связанные с линейками рёбрами BELONGS_TO."""
        graph = self._graph
        return {
            model
            for line in lines
            if line in graph
            for model, _, attrs in graph.in_edges(line, data=True)
            if attrs["type"] == "BELONGS_TO"
        }

    def _line_facts(self, models: set[str]) -> tuple[list[str], set[str], set[str]]:
        """Собирает итоги отказов и связи замен по моделям выбранных линеек."""
        graph = self._graph
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


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient, *, options: SearchOptions | None = None) -> Answer:
    """Отвечает на вопрос по связям графа вокруг опорных сущностей."""
    options = options if options is not None else SearchOptions()
    anchors = _extract_anchors(question, kb.vocabulary)
    context = KnowledgeSearch(kb).find_by_anchors(anchors, options)
    if context is None:
        return Answer(_NO_FACTS, refused=True)

    text = _answer_from_facts(question, context, llm, options.local_max_tokens)
    if _is_refusal(text):
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=context.sources, facts=context.facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient, *, options: SearchOptions | None = None) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага."""
    options = options if options is not None else SearchOptions()
    search = KnowledgeSearch(kb)
    line_names = _extract_line_names(question, kb.vocabulary)
    context = search.find_by_lines(line_names)
    partials: list[str] = []
    sources: set[str] = set()
    known_codes: set[str] = set()
    for summary in context.summaries:
        partial = _answer_from_summary(question, summary, context, llm, options)
        if partial is None:
            continue
        partials.append(partial)
        community_sources, community_codes = search.community_evidence(summary.community)
        sources.update(community_sources)
        known_codes.update(community_codes)

    if not partials:
        return Answer(_NO_FACTS, refused=True)

    text = _merge_answers(question, partials, context, llm, options.reduce_max_tokens)
    if _is_refusal(text) or fabricated_facts(text, question, known_codes | context.codes, tool_text=""):
        return Answer(_NO_FACTS, refused=True)
    return Answer(text, sources=sorted(sources | context.sources))


@dataclass(frozen=True)
class SearchAnchors:
    """ID опорных узлов, распознанных по кодам и по словарным названиям."""

    code_nodes: tuple[str, ...] = ()
    named_nodes: tuple[str, ...] = ()


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


@dataclass
class _LocalContext:
    facts: list[str]
    sources: list[str]
    single_part: bool


@dataclass
class _OverviewContext:
    facts: str
    sources: set[str]
    codes: set[str]
    has_scope: bool
    summaries: list[_ScopedSummary]


@dataclass
class _ScopedSummary:
    community: Community
    in_scope: bool


class _PartialAnswer(BaseModel):
    answer: str
    score: int = Field(strict=True, ge=0, le=10)


def _extract_anchors(question: str, vocabulary: dict[str, list[str]]) -> SearchAnchors:
    code_nodes = tuple(Entity(code_type(code), code).node_id for code in find_codes(question))
    named_nodes = tuple(Entity(kind, name).node_id for kind, name in find_names(question, vocabulary))
    return SearchAnchors(code_nodes=code_nodes, named_nodes=named_nodes)


def _answer_from_facts(question: str, context: _LocalContext, llm: LLMClient, max_tokens: int) -> str:
    system = _LOCAL_PROMPT
    if context.single_part:
        system += _LOCAL_PART_PROMPT
    facts = _numbered_lines(context.facts)
    user = f"Вопрос: {question}\n\nФакты:\n{facts}"
    return _ask(llm, system, user, max_tokens=max_tokens).strip()


def _is_refusal(text: str) -> bool:
    return not text or _NO_FACTS.rstrip(".").casefold() in text.casefold()


def _extract_line_names(question: str, vocabulary: dict[str, list[str]]) -> list[str]:
    product_lines = {"ProductLine": vocabulary.get("ProductLine", [])}
    return [name for _, name in find_names(question, product_lines)]


def _answer_from_summary(
    question: str, summary: _ScopedSummary, context: _OverviewContext, llm: LLMClient, options: SearchOptions
) -> str | None:
    system = _SCOPED_MAP_PROMPT if context.has_scope else _MAP_PROMPT
    user = f"Вопрос: {question}\n\nСводка:\n{summary.community.summary}"
    if not summary.in_scope:
        user += _OUT_OF_SCOPE
    elif context.facts:
        user += _LINE_FACTS_HEADER + context.facts + _MAP_LINE_PROMPT

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
    question: str, partials: list[str], context: _OverviewContext, llm: LLMClient, max_tokens: int
) -> str:
    answers = _numbered_lines(partials)
    system = _REDUCE_PROMPT
    if context.facts:
        answers += _LINE_FACTS_HEADER + context.facts
        system += _REDUCE_LINE_PROMPT
    elif len(context.summaries) == 1:
        summary = context.summaries[0].community.summary
        answers += f"\n\nСводка, подтверждающая частичный ответ:\n{summary}"
    user = f"Вопрос: {question}\n\nЧастичные ответы:\n{answers}"
    return _ask(llm, system, user, max_tokens=max_tokens).strip()


def _numbered_lines(lines: list[str]) -> str:
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


def _ask(llm: LLMClient, system: str, user: str, *, max_tokens: int, json_schema: dict | None = None) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return llm.chat(messages, json_schema=json_schema, max_tokens=max_tokens).content
