"""
Thin wrapper around the Azure AI Foundry embedding deployment. Same
endpoint/key as the chat deployment (backend/llm/client.py), different
deployment name — an embedding model has no chat/tool-calling behavior of
its own, so it gets its own minimal client rather than reusing chat's.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

from backend.observability import log_call

load_dotenv(Path(__file__).parent.parent / ".env")

EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


def _get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of strings in one API call — cheaper and faster than one call per text, and how the embeddings endpoint is meant to be used."""
    if not texts:
        return []
    client = _get_client()
    start = time.perf_counter()
    response = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=texts)
    log_call(
        call_type="embedding",
        deployment=EMBEDDING_DEPLOYMENT,
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=0,
        latency_ms=(time.perf_counter() - start) * 1000,
        caller="rag.embeddings.embed_texts",
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
