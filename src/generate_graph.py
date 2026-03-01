# GraphRAG-enhanced indexing for LlamaCloud

import os
import json
import logging
from typing import List
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from llama_index.core.readers import SimpleDirectoryReader
from tqdm import tqdm

from src.index import get_index
from src.service import LLamaCloudFileService
from src.settings import init_settings
from llama_index.core import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@dataclass
class Entity:
    name: str
    entity_type: str


@dataclass
class Relationship:
    source: str
    target: str
    relation_type: str


def extract_entities_and_relations(
    text: str,
) -> tuple[List[Entity], List[Relationship]]:
    """Extract entities and relationships from text using LLM."""

    prompt = f"""Extract entities and relationships from this text.

Return ONLY valid JSON:
{{
  "entities": [{{"name": "entity name", "type": "type"}}],
  "relationships": [{{"source": "A", "target": "B", "type": "relates to"}}]
}}

Text (first 2500 chars):
{text[:2500]}

JSON:"""

    try:
        response = Settings.llm.complete(prompt)
        content = response.text.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())

        entities = [
            Entity(name=e["name"], entity_type=e.get("type", "concept"))
            for e in data.get("entities", [])
        ]
        relationships = [
            Relationship(
                source=r["source"],
                target=r["target"],
                relation_type=r.get("type", "related_to"),
            )
            for r in data.get("relationships", [])
        ]

        return entities, relationships

    except Exception as e:
        logger.warning(f"Extraction failed: {e}")
        return [], []


def generate_graph_index():
    """Generate index with graph-enhanced metadata in LlamaCloud."""
    init_settings()

    logger.info("Starting graph-enhanced indexing...")

    index = get_index(create_if_missing=True)
    if index is None:
        raise ValueError("Could not create LlamaCloud index")

    reader = SimpleDirectoryReader("ui/data", recursive=True)
    files = reader.input_files

    if not files:
        logger.warning("No files in ui/data")
        return

    logger.info(f"Processing {len(files)} files...")

    all_entities = {}
    all_relationships = []

    for input_file in tqdm(files, desc="Processing files"):
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                continue

            # Extract graph data
            chunk_size = 2000
            chunks = [
                content[i : i + chunk_size]
                for i in range(0, len(content), chunk_size - 200)
            ]

            for chunk_idx, chunk in enumerate(chunks):
                entities, relationships = extract_entities_and_relations(chunk)

                for ent in entities:
                    key = ent.name.lower()
                    if key not in all_entities:
                        all_entities[key] = ent

                all_relationships.extend(relationships)

            # Add to LlamaCloud
            with open(input_file, "rb") as f:
                LLamaCloudFileService.add_file_to_pipeline(
                    index.project.id,
                    index.pipeline.id,
                    f,
                    custom_metadata={"graph_indexed": "true"},
                    wait_for_processing=False,
                )

        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")

    logger.info(f"Graph indexing complete!")
    logger.info(f"Total unique entities: {len(all_entities)}")
    logger.info(f"Total relationships: {len(all_relationships)}")

    if all_entities:
        sample = list(all_entities.values())[:10]
        logger.info("Sample entities:")
        for ent in sample:
            logger.info(f"  - {ent.name} ({ent.entity_type})")

    if all_relationships:
        logger.info("Sample relationships:")
        for rel in all_relationships[:10]:
            logger.info(f"  - {rel.source} --[{rel.relation_type}]--> {rel.target}")


if __name__ == "__main__":
    generate_graph_index()
