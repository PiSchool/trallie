import importlib

from trallie.providers import get_provider

from trallie.schema_generation.schema_generator import SchemaGenerator
from trallie.data_extraction.data_extractor import DataExtractor

_LAZY_PROVIDER_CLASSES = {
    "GroqProvider": "trallie.providers.groq",
    "OpenAIProvider": "trallie.providers.openai",
    "OllamaProvider": "trallie.providers.ollama",
    "HuggingFaceProvider": "trallie.providers.huggingface",
}


def __getattr__(name):
    # Keep `from trallie import GroqProvider` working without importing every SDK up front
    if name in _LAZY_PROVIDER_CLASSES:
        return getattr(importlib.import_module(_LAZY_PROVIDER_CLASSES[name]), name)
    raise AttributeError(f"module 'trallie' has no attribute {name!r}")
