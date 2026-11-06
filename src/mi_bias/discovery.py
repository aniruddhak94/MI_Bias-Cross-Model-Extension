"""Adapters around the repository's existing EAP graph and attribution code."""

from __future__ import annotations

from copy import deepcopy
from functools import partial


def rank_eap_edges(model, dataset, metric, top_k: int = 100):
    """Score the current Graph with EAP and return absolute-score-ranked edges."""
    from Helper_Functions.attribution import attribute
    from Helper_Functions.graph import Graph

    if top_k < 1:
        raise ValueError("top_k must be positive")
    graph = Graph.from_model(model)
    attribute(model, graph, dataset, partial(metric, loss=True, mean=True))
    ranked = sorted(
        graph.edges.items(),
        key=lambda item: abs(item[1].score) if item[1].score is not None else 0.0,
        reverse=True,
    )
    return graph, ranked[:top_k]


def verify_edge_ranking_with_patching(model, graph, dataset, metric, ranked_edges,
                                      evaluate_graph):
    """Evaluate nested top-edge circuits so approximate EAP ranks get causal checks."""
    results = []
    selected = set()
    for rank, (edge_id, edge) in enumerate(ranked_edges, start=1):
        selected.add(edge_id)
        candidate_graph = deepcopy(graph)
        for candidate_id, candidate in candidate_graph.edges.items():
            candidate.in_graph = candidate_id in selected
        candidate_graph.prune_dead_nodes()
        value = evaluate_graph(model, candidate_graph, dataset, metric).mean().item()
        results.append({"rank": rank, "edge_id": edge_id, "eap_score": edge.score,
                        "patched_metric": value})
    return results
