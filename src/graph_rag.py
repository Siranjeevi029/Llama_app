"""
GraphRAG module — builds and persists a PropertyGraphIndex from documents.
On first run it extracts entities/relations via the LLM and saves to disk.
Subsequent server starts load from disk (fast) and re-embed entity nodes
with local MiniLM (< 2 seconds, zero API calls).

RETRIEVAL STRATEGY
------------------
We use TWO sub-retrievers in the graph retriever:

1. LLMSynonymRetriever  — keyword / entity-name match
   Expands the query with synonyms via Groq, then searches entity names
   in the KG store.  Fast, great for named-entity queries ("What is VAD?").
   Works entirely via REST (no gRPC).

2. EmbeddingGraphRetriever — custom cosine-similarity retriever
   SimplePropertyGraphStore.vector_query() raises NotImplementedError
   (supports_vector_queries=False), so VectorContextRetriever never works.
   Instead we do cosine similarity ourselves over the entity node embeddings
   that _embed_entity_nodes() populates at startup.  Fires for paraphrase
   / semantic queries ("speech tasks", "audio models", "matryoshka").
"""

# ── Windows UTF-8 fix ──────────────────────────────────────────────────────
# LlamaIndex's SimpleKVStore uses open() without specifying encoding, which
# defaults to cp1252 on Windows and crashes on non-ASCII chars in documents.
# We patch builtins.open so ALL text-mode file opens default to UTF-8.
import builtins as _builtins

_original_open = _builtins.open


def _utf8_open(file, mode="r", *args, **kwargs):
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _original_open(file, mode, *args, **kwargs)


_builtins.open = _utf8_open
# ──────────────────────────────────────────────────────────────────────────

import logging
import math
import nest_asyncio
from pathlib import Path
from typing import List, Optional

nest_asyncio.apply()

from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core import QueryBundle
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    LLMSynonymRetriever,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

logger = logging.getLogger("uvicorn")

# Marker file written after a successful graph build so we can detect it on reload
_MARKER_FILE = "graph_index_built.ok"

# ── Custom extraction prompt ───────────────────────────────────────────────
# The default SimpleLLMPathExtractor stores everything with the generic
# label "entity", so the node name is lost.  This prompt forces the LLM
# to use the actual task/technology name as the entity identifier.
_EXTRACT_PROMPT = (
    "You are an expert knowledge-graph builder.\n"
    "Given the text below, extract up to {max_knowledge_triplets} subject->relation->object triplets.\n\n"
    "Rules:\n"
    "- Subject and Object MUST be specific, meaningful names — e.g. 'Speaker Diarization',\n"
    "  'Pipecat', 'VAD', 'TTS', 'GraphRAG', 'BitNet', 'MemU', task IDs like 'AI_N-148'.\n"
    "  NEVER use generic words like 'task', 'it', 'this', 'the model', 'the system'.\n"
    "- Relation should be a short verb phrase (e.g. 'is', 'uses', 'evaluates', 'enables').\n"
    "- Use the EXACT casing/spelling from the text for entity names.\n\n"
    "Text:\n"
    "{text}\n\n"
    "Return ONLY triplets, one per line, in the format below — no prose, no markdown:\n"
    "(subject, relation, object)\n"
    "(subject, relation, object)\n"
)


# ── Custom cosine-similarity graph retriever ───────────────────────────────
class EmbeddingGraphRetriever(BaseRetriever):
    """
    Retrieves graph entity nodes by cosine similarity between the query
    embedding and entity node embeddings set by _embed_entity_nodes().

    Bypasses VectorContextRetriever because SimplePropertyGraphStore raises
    NotImplementedError for vector_query() (supports_vector_queries=False).
    Instead we access pg_store.graph.nodes directly and compute cosine
    similarity in pure Python — no external dependencies, no API calls.
    """

    def __init__(
        self,
        index: PropertyGraphIndex,
        similarity_top_k: int = 6,
        similarity_cutoff: float = 0.3,
    ) -> None:
        self._index = index
        self._top_k = similarity_top_k
        self._cutoff = similarity_cutoff
        super().__init__()

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-10)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        # Embed the query with the same model used for entity nodes
        q_emb = Settings.embed_model.get_text_embedding(query_str)

        pg_store = self._index.property_graph_store
        graph_obj = getattr(pg_store, "graph", None)
        if graph_obj is None:
            logger.warning("[EmbeddingGraphRetriever] pg_store.graph is None")
            return []

        # ── Semantic cosine similarity pass ───────────────────────────────
        scored = []
        nodes_with_emb = 0
        for node in graph_obj.nodes.values():
            emb = getattr(node, "embedding", None)
            if emb is None:
                continue
            nodes_with_emb += 1
            score = self._cosine(q_emb, emb)
            if score >= self._cutoff:
                scored.append((score, node))

        logger.info(
            f"[EmbeddingGraphRetriever] query='{query_str}' | "
            f"embedded_nodes={nodes_with_emb} | above_cutoff={len(scored)}"
        )

        # ── Keyword / substring fallback for acronyms & exact names ───────
        # If no hits via embedding (e.g. "mcp" → "Model context protocol"),
        # fall back to checking whether the query tokens appear as a
        # contiguous substring of any entity name (case-insensitive).
        if not scored:
            q_lower = query_str.lower()
            # Strip common question words for cleaner matching
            for stop in ("what is", "what are", "tell me about", "explain", "describe"):
                q_lower = q_lower.replace(stop, "").strip()
            q_tokens = [t for t in q_lower.split() if len(t) > 1]

            for node in graph_obj.nodes.values():
                name = (getattr(node, "name", None) or getattr(node, "text", "") or "").lower()
                if not name:
                    continue
                # Match if ALL query tokens appear in the entity name OR
                # if the entity name contains any query token as a substring
                if any(tok in name for tok in q_tokens):
                    emb = getattr(node, "embedding", None)
                    if emb is not None:
                        sim = self._cosine(q_emb, emb)
                        scored.append((max(sim, 0.15), node))  # floor at 0.15

            if scored:
                logger.info(
                    f"[EmbeddingGraphRetriever] keyword fallback found {len(scored)} node(s) "
                    f"for '{q_lower}'"
                )

        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Build results: prefer source text chunks over bare entity content ──
        # The graph docstore holds the actual document chunk text nodes that
        # the LLM path extractor processed.  Returning those gives the LLM
        # rich context instead of sparse "Entity --[Rel]--> Entity" lines.
        docstore = getattr(self._index, "docstore", None) or getattr(
            self._index, "_docstore", None
        )

        # Pre-index text_chunk nodes from the property graph (fallback)
        chunk_nodes = {
            nid: n
            for nid, n in graph_obj.nodes.items()
            if getattr(n, "label", "") == "text_chunk"
        }

        results: List[NodeWithScore] = []
        seen_ids: set = set()

        for score, entity_node in scored[: self._top_k]:
            entity_name = getattr(entity_node, "name", None) or getattr(entity_node, "text", "") or ""

            # 1. Try docstore: find chunk(s) whose text contain the entity name
            node_added = False
            if docstore is not None:
                try:
                    for doc_id, doc_node in docstore.docs.items():
                        if doc_id in seen_ids:
                            continue
                        chunk_text = getattr(doc_node, "text", "") or ""
                        if entity_name and entity_name.lower() in chunk_text.lower():
                            seen_ids.add(doc_id)
                            results.append(
                                NodeWithScore(node=doc_node, score=score)
                            )
                            node_added = True
                            break  # one source chunk per entity match
                except Exception:
                    pass

            # 2. Fallback: try in-graph text_chunk nodes
            if not node_added:
                for chunk_id, chunk_node in chunk_nodes.items():
                    if chunk_id in seen_ids:
                        continue
                    chunk_text = getattr(chunk_node, "text", "") or ""
                    if entity_name and entity_name.lower() in chunk_text.lower():
                        seen_ids.add(chunk_id)
                        text_node = TextNode(id_=chunk_id, text=chunk_text)
                        results.append(NodeWithScore(node=text_node, score=score))
                        node_added = True
                        break

            # 3. Last resort: return the entity node itself with triplets
            if not node_added and entity_node.id not in seen_ids:
                seen_ids.add(entity_node.id)
                triplets = pg_store.get_triplets(entity_names=[entity_node.id])
                triplet_lines = [f"{t[0].id} --[{t[1].id}]--> {t[2].id}" for t in triplets]
                content = entity_name
                if triplet_lines:
                    content += "\n" + "\n".join(triplet_lines)
                text_node = TextNode(id_=entity_node.id, text=content)
                results.append(NodeWithScore(node=text_node, score=score))

        logger.info(f"[EmbeddingGraphRetriever] returning {len(results)} node(s)")
        return results

def _embed_entity_nodes(index: PropertyGraphIndex) -> None:
    """
    Embed property graph ENTITY nodes using the current embed model so that
    EmbeddingGraphRetriever can do cosine similarity search on them.

    SimplePropertyGraphStore.vector_query() raises NotImplementedError —
    VectorContextRetriever is completely broken for this backend.
    Instead EmbeddingGraphRetriever reads node.embedding directly.

    This function embeds all entity nodes (non-text_chunk) at startup using
    local MiniLM — < 2 seconds, zero API calls.
    """
    pg_store = index.property_graph_store

    # SimplePropertyGraphStore stores nodes in self.graph (LabelledPropertyGraph)
    # which has a .nodes dict keyed by node id.
    graph_obj = getattr(pg_store, "graph", None)
    if graph_obj is None:
        logger.warning("[GraphRAG] Cannot access pg_store.graph — skipping entity embedding")
        return

    nodes_dict = getattr(graph_obj, "nodes", None)
    if not nodes_dict:
        logger.warning("[GraphRAG] pg_store.graph.nodes is empty — skipping entity embedding")
        return

    all_nodes = list(nodes_dict.values())
    # Only embed non-text_chunk nodes (the actual KG entities)
    entity_nodes = [n for n in all_nodes if getattr(n, "label", "") != "text_chunk"]
    if not entity_nodes:
        logger.warning("[GraphRAG] No entity nodes found to embed")
        return

    texts = [getattr(n, "text", None) or getattr(n, "name", None) or "" for n in entity_nodes]
    logger.info(f"[GraphRAG] Embedding {len(entity_nodes)} entity nodes with MiniLM...")
    embeddings = Settings.embed_model.get_text_embedding_batch(texts, show_progress=True)
    for node, emb in zip(entity_nodes, embeddings):
        node.embedding = emb
    # upsert back so the embeddings live on the in-memory node objects
    pg_store.upsert_nodes(entity_nodes)
    logger.info("[GraphRAG] Entity node embeddings ready ✓")


def build_or_load_graph_index(
    data_dir: str = "ui/data",
    persist_dir: str = "graph_index",
) -> Optional[PropertyGraphIndex]:
    """
    Returns a PropertyGraphIndex — loading from disk if already built,
    otherwise building from scratch and persisting to disk.
    """
    persist_path = Path(persist_dir)
    marker = persist_path / _MARKER_FILE

    # ── Try loading from disk ──────────────────────────────────────────────
    if persist_path.exists() and marker.exists():
        logger.info(f"[GraphRAG] Loading PropertyGraphIndex from '{persist_dir}'...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info("[GraphRAG] PropertyGraphIndex loaded from disk ✓")
            # Re-embed entity nodes every startup — fast with local MiniLM,
            # required because SimplePropertyGraphStore doesn't restore these
            # embeddings from disk (vector_query is NotImplemented on this backend).
            _embed_entity_nodes(index)
            return index
        except Exception as e:
            logger.warning(
                f"[GraphRAG] Failed to load from disk ({e}). Rebuilding..."
            )

    # ── Build fresh ────────────────────────────────────────────────────────
    logger.info(f"[GraphRAG] Building PropertyGraphIndex from '{data_dir}'...")

    try:
        reader = SimpleDirectoryReader(data_dir, recursive=True)
        documents = reader.load_data()
    except Exception as e:
        logger.error(f"[GraphRAG] Could not read documents: {e}")
        return None

    if not documents:
        logger.warning(f"[GraphRAG] No documents found in '{data_dir}'")
        return None

    logger.info(f"[GraphRAG] Extracting graph from {len(documents)} document(s)...")

    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=15,  # more paths -> richer graph
        num_workers=1,           # sequential to avoid hammering the API
        extract_prompt=_EXTRACT_PROMPT,
    )

    try:
        index = PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=[kg_extractor],
            embed_model=Settings.embed_model,
            show_progress=True,
        )
    except Exception as e:
        logger.error(f"[GraphRAG] Failed to build PropertyGraphIndex: {e}")
        return None

    # Embed entity nodes explicitly after build
    _embed_entity_nodes(index)

    # Persist to disk and write the marker file
    try:
        persist_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        marker.touch()
        logger.info(f"[GraphRAG] PropertyGraphIndex saved to '{persist_dir}' ✓")
    except Exception as e:
        logger.warning(f"[GraphRAG] Could not persist graph index: {e}")

    return index


def get_graph_retriever(
    index: PropertyGraphIndex,
    similarity_top_k: int = 6,
) -> BaseRetriever:
    """
    Returns a graph retriever fusing two complementary sub-retrievers:

    1. LLMSynonymRetriever — keyword / entity-name match via Groq
       Best for: "What is Pipecat?", "Tell me about VAD", named-entity queries.

    2. EmbeddingGraphRetriever — cosine-similarity over entity node embeddings
       Best for: "matryoshka", "speech tasks", broad/paraphrase queries.
       Replaces VectorContextRetriever which is broken for SimplePropertyGraphStore.
    """
    sub_retrievers = []

    # ── Sub-retriever 1: keyword synonym search ────────────────────────────
    sub_retrievers.append(
        LLMSynonymRetriever(
            index.property_graph_store,
            llm=Settings.llm,
            include_text=True,
            synonym_num=20,
            max_keywords=20,
            output_parsing_fn=None,
        )
    )

    # ── Sub-retriever 2: cosine similarity over embedded entity nodes ──────
    # cutoff=0.5: only high-confidence entity matches get returned.
    # This prevents broad queries like "list all tasks" (entity scores ~0.3)
    # from pulling in sparse entity nodes that confuse the LLM.
    # The keyword fallback in EmbeddingGraphRetriever still handles acronyms.
    emb_retriever = EmbeddingGraphRetriever(
        index,
        similarity_top_k=similarity_top_k,
        similarity_cutoff=0.5,
    )
    sub_retrievers.append(emb_retriever)
    logger.info("[GraphRAG] EmbeddingGraphRetriever enabled ✓")

    return index.as_retriever(
        sub_retrievers=sub_retrievers,
        include_text=True,
    )
