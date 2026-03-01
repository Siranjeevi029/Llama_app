"""
HybridRetriever — fuses results from up to 3 retrievers using
Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = Σ  1 / (k + rank(d))   [k=60, standard constant]

Why RRF instead of weighted average?
- Vector, BM25, and graph scores live on incompatible scales.
- RRF only uses rank positions, so it is scale-agnostic and robust.
"""

import logging
from typing import List, Optional

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger("uvicorn")

# Standard RRF constant — reduces dominance of top-1 results
_RRF_K = 60


class HybridRetriever(BaseRetriever):
    """
    Combines dense (vector), sparse (BM25), and graph retrievers
    using Reciprocal Rank Fusion.
    All three retrievers are optional — at least one must be provided.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: Optional[BaseRetriever] = None,
        graph_retriever: Optional[BaseRetriever] = None,
        top_n: int = 10,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._graph_retriever = graph_retriever
        self._top_n = top_n
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # ── Collect results from each retriever ───────────────────────────
        retriever_results: List[List[NodeWithScore]] = []

        # 1. Vector retriever (always present)
        try:
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            retriever_results.append(vector_nodes)
            logger.info(f"[Hybrid] Vector: {len(vector_nodes)} nodes")
        except Exception as e:
            logger.warning(f"[Hybrid] Vector retriever failed: {e}")
            retriever_results.append([])

        # 2. BM25 retriever (optional)
        if self._bm25_retriever is not None:
            try:
                bm25_nodes = self._bm25_retriever.retrieve(query_bundle)
                retriever_results.append(bm25_nodes)
                logger.info(f"[Hybrid] BM25: {len(bm25_nodes)} nodes")
            except Exception as e:
                logger.warning(f"[Hybrid] BM25 retriever failed: {e}")
                retriever_results.append([])

        # 3. Graph retriever (optional)
        if self._graph_retriever is not None:
            try:
                graph_nodes = self._graph_retriever.retrieve(query_bundle)
                retriever_results.append(graph_nodes)
                logger.info(f"[Hybrid] Graph: {len(graph_nodes)} nodes")
            except Exception as e:
                logger.warning(f"[Hybrid] Graph retriever failed: {e}")
                retriever_results.append([])

        # ── Reciprocal Rank Fusion ─────────────────────────────────────────
        rrf_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for result_list in retriever_results:
            for rank, node_with_score in enumerate(result_list):
                nid = node_with_score.node.node_id
                rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (_RRF_K + rank + 1)
                node_map[nid] = node_with_score

        # Sort descending by RRF score and return top-N
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        fused = [
            NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid])
            for nid in sorted_ids[: self._top_n]
        ]

        logger.info(f"[Hybrid] RRF fused → {len(fused)} final nodes")
        return fused
