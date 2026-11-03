"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

from itertools import chain

import networkx as nx

from assistant.infra import schemas


def _add_entity(graph: nx.DiGraph, entity: schemas.Entity, chunk_id: str) -> None:
    if entity.node_id not in graph:
        graph.add_node(
            entity.node_id, type=entity.type, name=entity.name, sources=set()
        )
    graph.nodes[entity.node_id]["sources"].add(chunk_id)


def _resolve_replacement(graph: nx.DiGraph, subject: str, object_: str) -> None:
    forward = graph.edges[subject, object_]
    reverse = graph.get_edge_data(object_, subject)
    if reverse is None or reverse["type"] != "REPLACED_BY":
        return

    graph.graph["conflicts"] += 1
    if forward["weight"] <= reverse["weight"]:
        graph.remove_edge(subject, object_)
    if reverse["weight"] <= forward["weight"]:
        graph.remove_edge(object_, subject)


def build_graph(results: list[schemas.ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)
    relations = [
        (relation, result.chunk_id)
        for result in results
        for relation in result.relations
        if relation.matches_schema()
    ]
    entities = ((entity, result.chunk_id) for result in results for entity in result.entities)
    endpoints = (
        (entity, chunk_id)
        for relation, chunk_id in relations
        for entity in (relation.subject, relation.object)
    )

    for entity, chunk_id in chain(entities, endpoints):
        _add_entity(graph, entity, chunk_id)

    for relation, chunk_id in relations:
        subject, object_ = relation.subject.node_id, relation.object.node_id
        if not graph.has_edge(subject, object_):
            graph.add_edge(subject, object_, type=relation.type, sources=set())
        graph.edges[subject, object_]["sources"].add(chunk_id)

    for _, attributes in graph.nodes(data=True):
        attributes["sources"] = sorted(attributes["sources"])
    for _, _, attributes in graph.edges(data=True):
        attributes["sources"] = sorted(attributes["sources"])
        attributes["weight"] = len(attributes["sources"])

    # Снимок рёбер позволяет удалять их при обходе; порядок исключает повтор пары.
    for subject, object_, attributes in list(graph.edges(data=True)):
        if subject < object_ and attributes["type"] == "REPLACED_BY":
            _resolve_replacement(graph, subject, object_)

    return graph
