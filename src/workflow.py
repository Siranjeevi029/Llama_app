import os
import logging

from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    SimpleKeywordTableIndex,
    VectorStoreIndex,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from src.custom_retriever import CustomRetriever
from src.query import get_query_engine_tool
from src.citation import CITATION_SYSTEM_PROMPT, enable_citation
from src.settings import init_settings

logger = logging.getLogger("uvicorn")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "ui", "data")


def create_workflow() -> AgentWorkflow:
    load_dotenv()
    init_settings()

    # Load documents from the data directory
    if not os.path.exists(DATA_DIR):
        raise RuntimeError(
            f"Data directory '{DATA_DIR}' not found! Please add documents to ui/data/."
        )
    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
    if not documents:
        raise RuntimeError(
            "No documents found in ui/data/. Please add documents first."
        )

    # Parse documents into nodes and build a shared storage context
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # Build both indexes from the same nodes
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

    # Create retrievers
    top_k = int(os.getenv("TOP_K", "3"))
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=top_k)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)

    # Combine into a hybrid custom retriever (OR = union of results)
    custom_retriever = CustomRetriever(
        vector_retriever, keyword_retriever, mode="OR"
    )

    logger.info(
        f"Built hybrid retriever with {len(nodes)} nodes from {len(documents)} documents"
    )

    # Create a query tool using the custom retriever, with citations enabled
    query_tool = enable_citation(get_query_engine_tool(retriever=custom_retriever))

    # Define the system prompt for the agent
    system_prompt = """You are a helpful assistant"""
    system_prompt += CITATION_SYSTEM_PROMPT

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_tool],
        llm=Settings.llm,
        system_prompt=system_prompt,
    )


workflow = create_workflow()
