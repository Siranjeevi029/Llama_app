import logging
import os
from typing import Optional

from llama_cloud import PipelineType
from llama_index.core.callbacks import CallbackManager
from llama_index.core.ingestion.api_utils import (
    get_client as llama_cloud_get_client,
)
from llama_index.core.settings import Settings
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("uvicorn")


class LlamaCloudConfig(BaseModel):
    # Private attributes
    api_key: str = Field(
        exclude=True,  # Exclude from the model representation
    )
    base_url: Optional[str] = Field(
        exclude=True,
    )
    organization_id: Optional[str] = Field(
        exclude=True,
    )
    # Configuration attributes, can be set by the user
    pipeline: str = Field(
        description="The name of the pipeline to use",
    )
    project: str = Field(
        description="The name of the LlamaCloud project",
    )

    def __init__(self, **kwargs):
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("LLAMA_CLOUD_API_KEY")
        if "base_url" not in kwargs:
            kwargs["base_url"] = os.getenv("LLAMA_CLOUD_BASE_URL")
        if "organization_id" not in kwargs:
            kwargs["organization_id"] = os.getenv("LLAMA_CLOUD_ORGANIZATION_ID")
        if "pipeline" not in kwargs:
            kwargs["pipeline"] = os.getenv("LLAMA_CLOUD_INDEX_NAME")
        if "project" not in kwargs:
            kwargs["project"] = os.getenv("LLAMA_CLOUD_PROJECT_NAME")
        super().__init__(**kwargs)

    # Validate and throw error if the env variables are not set before starting the app
    @field_validator("pipeline", "project", "api_key", mode="before")
    @classmethod
    def validate_fields(cls, value):
        if value is None:
            raise ValueError(
                "Please set LLAMA_CLOUD_INDEX_NAME, LLAMA_CLOUD_PROJECT_NAME and LLAMA_CLOUD_API_KEY"
                " to your environment variables or config them in .env file"
            )
        return value

    def to_client_kwargs(self) -> dict:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }


class IndexConfig(BaseModel):
    llama_cloud_pipeline_config: LlamaCloudConfig = Field(
        default_factory=lambda: LlamaCloudConfig(),
        alias="llamaCloudPipeline",
    )
    callback_manager: Optional[CallbackManager] = Field(
        default=None,
    )

    def to_index_kwargs(self) -> dict:
        return {
            "name": self.llama_cloud_pipeline_config.pipeline,
            "project_name": self.llama_cloud_pipeline_config.project,
            "api_key": self.llama_cloud_pipeline_config.api_key,
            "base_url": self.llama_cloud_pipeline_config.base_url,
            "organization_id": self.llama_cloud_pipeline_config.organization_id,
            "callback_manager": self.callback_manager,
        }


def get_index(
    config: IndexConfig = None,
    create_if_missing: bool = False,
):
    if config is None:
        config = IndexConfig()
    # Check whether the index exists
    try:
        index = LlamaCloudIndex(**config.to_index_kwargs())
        return index
    except ValueError:
        logger.warning("Index not found")
        if create_if_missing:
            logger.info("Creating index")
            _create_index(config)
            return LlamaCloudIndex(**config.to_index_kwargs())
        return None


def get_client():
    config = LlamaCloudConfig()
    return llama_cloud_get_client(**config.to_client_kwargs())


def _create_index(
    config: IndexConfig,
):
    client = get_client()
    pipeline_name = config.llama_cloud_pipeline_config.pipeline

    pipelines = client.pipelines.search_pipelines(
        pipeline_name=pipeline_name,
        pipeline_type=PipelineType.MANAGED.value,
    )
    if len(pipelines) == 0:
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.embeddings.gemini import GeminiEmbedding

        embedding_type = None
        embedding_component = {}
        if isinstance(Settings.embed_model, OpenAIEmbedding):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            embedding_type = "OPENAI_EMBEDDING"
            embedding_component = {
                "api_key": openai_api_key,
                "model_name": os.getenv("EMBEDDING_MODEL"),
            }
        elif isinstance(Settings.embed_model, GeminiEmbedding):
            gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_api_key is None:
                raise ValueError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini embeddings."
                )
            embedding_type = "GEMINI_EMBEDDING"
            embedding_component = {
                "api_key": gemini_api_key,
                "model_name": os.getenv("EMBEDDING_MODEL"),
            }
        else:
            raise ValueError(
                "Auto-creating a pipeline from this app currently supports OpenAI and Gemini embeddings. "
                "For other embedding providers, create the pipeline in LlamaCloud first and set "
                "LLAMA_CLOUD_INDEX_NAME to that existing pipeline."
            )
        client.pipelines.upsert_pipeline(
            request={
                "name": pipeline_name,
                "embedding_config": {
                    "type": embedding_type,
                    "component": embedding_component,
                },
                "transform_config": {
                    "mode": "auto",
                    "config": {
                        "chunk_size": Settings.chunk_size,  # editable
                        "chunk_overlap": Settings.chunk_overlap,  # editable
                    },
                },
            },
        )
