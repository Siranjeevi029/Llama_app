# GraphRAG using LlamaIndex PropertyGraphIndex

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Document
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import TextNode
from llama_index.core.query_engine import RetrieverQueryEngine

from tqdm import tqdm

from src.index import get_index
from src.service import LLamaCloudFileService
from src.settings import init_settings
from llama_index.core import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_property_graph_index():
    """Generate PropertyGraphIndex with extraction."""
    init_settings()

    logger.info("=" * 60)
    logger.info("Starting PropertyGraphIndex Generation...")
    logger.info("=" * 60)

    # Load documents
    reader = SimpleDirectoryReader("ui/data", recursive=True)
    files = reader.input_files

    if not files:
        logger.warning("No files in ui/data")
        return

    logger.info(f"Found {len(files)} files")

    # Read all document content
    all_texts = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = file.read()
                if content.strip():
                    all_texts.append(content)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    if not all_texts:
        logger.error("No text content to process")
        return

    # Combine all texts
    full_text = "\n\n".join(all_texts)

    # Split into chunks
    chunk_size = 2000
    chunks = [
        full_text[i : i + chunk_size]
        for i in range(0, len(full_text), chunk_size - 200)
    ]

    logger.info(f"Split into {len(chunks)} chunks")

    # Create documents for PropertyGraphIndex
    documents = [Document(text=chunk) for chunk in chunks]

    # Use simple graph extractor
    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
    )

    logger.info("Building PropertyGraphIndex...")

    # Create in-memory PropertyGraphIndex
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        embed_model=Settings.embed_model,
        show_progress=True,
    )

    logger.info("PropertyGraphIndex created!")

    # Add files to LlamaCloud as well
    main_index = get_index(create_if_missing=True)
    if main_index:
        for f in files:
            try:
                with open(f, "rb") as file:
                    LLamaCloudFileService.add_file_to_pipeline(
                        main_index.project.id,
                        main_index.pipeline.id,
                        file,
                        custom_metadata={"property_graph_indexed": "true"},
                        wait_for_processing=False,
                    )
            except Exception as e:
                logger.error(f"Error adding to LlamaCloud: {e}")

    logger.info("=" * 60)
    logger.info("PropertyGraphIndex generation complete!")
    logger.info("=" * 60)

    return index


def create_graph_rag_workflow():
    """Create the full GraphRAG workflow with PropertyGraphIndex."""
    init_settings()

    logger.info("Building GraphRAG workflow...")

    # Generate or load the property graph index
    try:
        index = generate_property_graph_index()
    except Exception as e:
        logger.error(f"Failed to create PropertyGraphIndex: {e}")
        return None

    # Create retriever
    retriever = index.as_retriever(
        similarity_top_k=5,
        include_text=True,
    )

    # Create query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=Settings.llm,
    )

    logger.info("GraphRAG workflow ready!")

    return {
        "index": index,
        "retriever": retriever,
        "query_engine": query_engine,
    }


if __name__ == "__main__":
    result = create_graph_rag_workflow()
    if result:
        logger.info("Done! Use the query_engine to answer questions.")
