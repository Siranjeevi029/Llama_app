import os
from pathlib import Path

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from dotenv import load_dotenv

env_path = Path(__file__).parents[1] / ".env"
load_dotenv(env_path)


def init_settings():
    if os.getenv("GROQ_API_KEY") is None:
        raise RuntimeError("GROQ_API_KEY is missing in environment variables")

    llm_timeout = float(os.getenv("LLM_TIMEOUT_SEC", "45"))
    llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))  # default 2000 for full responses
    Settings.llm = Groq(
        model=os.getenv("MODEL") or "llama-3.3-70b-versatile",
        timeout=llm_timeout,
        max_retries=llm_max_retries,
        max_tokens=llm_max_tokens,
    )
    # Use a local sentence-transformers model — zero API calls, no rate limits.
    # all-MiniLM-L6-v2: 384-dim, ~90 MB on disk, fast CPU inference.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2",
    )
