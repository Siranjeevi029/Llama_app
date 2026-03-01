import os
import logging
from typing import Any, List

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

from src.index import get_index
from src.settings import init_settings
from src.custom_retriever import HybridRetriever
from src.graph_rag import build_or_load_graph_index, get_graph_retriever

logger = logging.getLogger("uvicorn")

init_settings()

# ── 1. Connect to LlamaCloud index ────────────────────────────────────────
index = get_index()
if index is None:
    raise RuntimeError("Could not connect to LlamaCloud index.")
logger.info("Connected to LlamaCloud index ✓")

top_k = int(os.getenv("TOP_K", "6"))

# ── 2. BM25 retriever — initialised from a bulk LlamaCloud fetch ──────────
bm25_retriever = None
all_nodes = []

try:
    logger.info("Fetching nodes from LlamaCloud for BM25...")
    _bootstrap_retriever = index.as_retriever(similarity_top_k=100)
    all_results = _bootstrap_retriever.retrieve("document")
    all_nodes = [n.node for n in all_results]
    logger.info(f"Fetched {len(all_nodes)} nodes for BM25")

    for i, node in enumerate(all_nodes[:3]):
        try:
            logger.info(f"  Node {i + 1} text length: {len(node.text)} chars")
        except Exception:
            pass

    if all_nodes:
        # Clamp BM25 k to corpus size — BM25Retriever errors when k > len(nodes)
        bm25_top_k = min(top_k, len(all_nodes))
        bm25_retriever = BM25Retriever(nodes=all_nodes, similarity_top_k=bm25_top_k)
        logger.info(f"BM25 retriever ready ({len(all_nodes)} nodes, k={bm25_top_k}) ✓")
except Exception as e:
    logger.warning(f"Could not initialise BM25 retriever: {e}")

# ── 3. GraphRAG retriever — build/load PropertyGraphIndex ─────────────────
graph_retriever = None
try:
    logger.info("Initialising GraphRAG retriever...")
    graph_index = build_or_load_graph_index(
        data_dir="ui/data",
        persist_dir="graph_index",
    )
    if graph_index is not None:
        graph_retriever = get_graph_retriever(graph_index, similarity_top_k=top_k)
        logger.info("GraphRAG retriever ready ✓")
    else:
        logger.warning("GraphRAG unavailable — no graph index built.")
except Exception as e:
    logger.warning(f"Could not initialise GraphRAG retriever: {e}")

# ── 4. Assemble the query engine ──────────────────────────────────────────
vector_retriever = index.as_retriever(similarity_top_k=top_k)

_active_modes = ["Vector"]
if bm25_retriever:
    _active_modes.append("BM25")
if graph_retriever:
    _active_modes.append("Graph")

try:
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        graph_retriever=graph_retriever,
        top_n=top_k * 2,
    )
    response_synthesizer = get_response_synthesizer(llm=Settings.llm)
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
    )
    logger.info(f"Query engine ready — mode: {' + '.join(_active_modes)} ✓")
except Exception as e:
    logger.warning(f"Could not create hybrid query engine: {e}. Falling back to vector-only.")
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    logger.info("Query engine ready — mode: Vector-only (fallback)")


# ── 5. QueryWorkflow (entry point for llama_deploy) ───────────────────────
class QueryWorkflow:
    def __init__(self):
        self.query_engine = query_engine
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.graph_retriever = graph_retriever
        self.all_nodes = all_nodes

    async def run(self, **kwargs) -> str:
        message = kwargs.get("message", kwargs.get("input", ""))
        if not message and kwargs:
            message = str(list(kwargs.values())[0]) if kwargs else ""

        logger.info(f"Query received: {message}")
        try:
            response = await self.query_engine.aquery(message)
            result = str(response)
            logger.info(f"Result preview ({len(result)} chars total): {result[:500]}{'...' if len(result) > 500 else ''}")
            return result
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error: {str(e)}"


workflow = QueryWorkflow()
