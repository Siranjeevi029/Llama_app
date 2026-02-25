import os
from typing import Any, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core import get_response_synthesizer


def create_query_engine(
    index: Optional[BaseIndex] = None,
    retriever: Optional[BaseRetriever] = None,
    **kwargs: Any,
) -> BaseQueryEngine:
    """
    Create a query engine from an index or a custom retriever.

    Args:
        index: The index to create a query engine for.
        retriever: A custom retriever to use instead of the index default.
        params (optional): Additional parameters for the query engine, e.g: similarity_top_k
    """
    if retriever is not None:
        response_synthesizer = get_response_synthesizer()
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

    if index is None:
        raise ValueError("Either index or retriever must be provided.")

    top_k = int(os.getenv("TOP_K", 0))
    if top_k != 0 and kwargs.get("filters") is None:
        kwargs["similarity_top_k"] = top_k

    return index.as_query_engine(**kwargs)


def get_query_engine_tool(
    index: Optional[BaseIndex] = None,
    retriever: Optional[BaseRetriever] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> QueryEngineTool:
    """
    Get a query engine tool for the given index or custom retriever.

    Args:
        index: The index to create a query engine for.
        retriever: A custom retriever to use instead of the index default.
        name (optional): The name of the tool.
        description (optional): The description of the tool.
    """
    if name is None:
        name = "query_index"
    if description is None:
        description = "Use this tool to retrieve information from a knowledge base. Provide a specific query and can call the tool multiple times if necessary."
    query_engine = create_query_engine(index=index, retriever=retriever, **kwargs)
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=name,
        description=description,
    )
    return tool
