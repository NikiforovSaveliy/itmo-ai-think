"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает взвешенный граф на сообщества алгоритмом Лувена."""
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for start, end, attrs in sorted(graph.edges(data=True)):
        previous = undirected.get_edge_data(start, end, default={})
        undirected.add_edge(start, end, weight=previous.get("weight", 0) + attrs.get("weight", 1))

    if not undirected.size(weight="weight"):
        return [Community(index, [node]) for index, node in enumerate(undirected)]

    groups = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    return [Community(index, nodes) for index, nodes in enumerate(sorted(map(sorted, groups)))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Описывает факты сообщества и рассчитанную статистику за 150 слов."""
    members = ", ".join(graph.nodes[node]["name"] for node in community.nodes)
    facts: list[str] = []
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for start, end, attrs in sorted(graph.out_edges(community.nodes, data=True)):
        subject = graph.nodes[start]["name"]
        target = graph.nodes[end]["name"]
        relation = attrs["type"]
        weight = attrs.get("weight", 1)
        facts.append(f"{subject} -{relation}-> {target}: {weight}")
        counts[relation] += weight
        if relation == "FAILED":
            failures[target] += weight

    totals = ", ".join(f"{relation}: {count}" for relation, count in sorted(counts.items()))
    failure_totals = ", ".join(f"{component}: {count}" for component, count in failures.most_common())
    fact_text = "\n".join(facts)
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши сводку сообщества по-русски, до 150 слов. "
                    "Опиши его сущности, связи и частые отказы. Сохрани коды, названия и готовые числа. "
                    "Для отказов укажи, к каким моделям относятся числа. Используй только переданные факты."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Сущности: {members}\n"
                    f"Суммы весов по типам связей: {totals or 'нет'}\n"
                    f"Отказы по узлам: {failure_totals or 'нет'}\n"
                    f"Факты и веса рёбер:\n{fact_text}"
                ),
            },
        ]
    )
    return " ".join(response.content.split()[:150])
