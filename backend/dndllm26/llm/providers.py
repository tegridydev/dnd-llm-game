from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from dndllm26.core.settings import Settings


@dataclass(frozen=True, slots=True)
class Preset:
    label: str
    url: str
    protocol: str = "openai"
    key_required: bool = False


PRESETS = {
    "ollama": Preset("Ollama", "http://127.0.0.1:11434", "ollama"),
    "lmstudio": Preset("LM Studio", "http://127.0.0.1:1234/v1"),
    "llamacpp": Preset("llama.cpp", "http://127.0.0.1:8080/v1"),
    "vllm": Preset("vLLM", "http://127.0.0.1:8000/v1"),
    "localai": Preset("LocalAI", "http://127.0.0.1:8080/v1"),
    "koboldcpp": Preset("KoboldCpp", "http://127.0.0.1:5001/v1"),
    "openai": Preset("OpenAI", "https://api.openai.com/v1", key_required=True),
    "anthropic": Preset("Claude", "https://api.anthropic.com/v1", "anthropic", True),
    "gemini": Preset(
        "Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", key_required=True
    ),
    "openrouter": Preset("OpenRouter", "https://openrouter.ai/api/v1", key_required=True),
    "groq": Preset("Groq", "https://api.groq.com/openai/v1", key_required=True),
    "mistral": Preset("Mistral", "https://api.mistral.ai/v1", key_required=True),
    "deepseek": Preset("DeepSeek", "https://api.deepseek.com/v1", key_required=True),
    "xai": Preset("xAI", "https://api.x.ai/v1", key_required=True),
    "together": Preset("Together AI", "https://api.together.xyz/v1", key_required=True),
    "fireworks": Preset("Fireworks AI", "https://api.fireworks.ai/inference/v1", key_required=True),
    "cerebras": Preset("Cerebras", "https://api.cerebras.ai/v1", key_required=True),
    "perplexity": Preset("Perplexity", "https://api.perplexity.ai", key_required=True),
    "huggingface": Preset(
        "Hugging Face Router", "https://router.huggingface.co/v1", key_required=True
    ),
    "azure": Preset("Azure OpenAI", "", key_required=True),
    "compatible": Preset("OpenAI-compatible", ""),
}

MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,159}")


def model_id(value: str) -> str:
    value = value.strip()
    if not MODEL_ID.fullmatch(value):
        raise ValueError(
            "Enter a model ID using up to 160 letters, digits or model-name characters."
        )
    return value


@dataclass(frozen=True, slots=True)
class Endpoint:
    name: str
    label: str
    protocol: str
    base_url: str
    remote: bool
    key: str = field(default="", repr=False)

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.base_url}"

    @property
    def headers(self) -> dict[str, str]:
        if self.protocol == "anthropic":
            return {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        if self.name == "azure":
            return {"api-key": self.key}
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    chat: Endpoint
    embeddings: Endpoint | None
    chat_model: str
    utility_model: str
    embed_model: str


def _endpoint(name: str, url: str, key: str, *, remote_allowed: bool) -> Endpoint:
    if name not in PRESETS:
        raise ValueError(f"Unknown model provider. Choose from {', '.join(PRESETS)}.")
    preset = PRESETS[name]
    url = url.strip().rstrip("/")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        port = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or port == 0
            or parsed.username is not None
            or parsed.password is not None
            or re.search(r"[\x00-\x20\x7f\\?#]", url)
        ):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "Set a valid model base URL without credentials, queries or fragments."
        ) from exc
    if host == "localhost":
        host = "127.0.0.1"
        parsed = parsed._replace(netloc="127.0.0.1" + (f":{port}" if port else ""))
    try:
        remote = not ipaddress.ip_address(host).is_loopback
    except ValueError:
        remote = True
    if remote and (parsed.scheme != "https" or not remote_allowed):
        raise ValueError(
            "Remote model endpoints require HTTPS and the corresponding remote opt-in."
        )
    if name == "azure" and (
        not host.endswith((".openai.azure.com", ".services.ai.azure.com"))
        or not parsed.path.endswith("/openai/v1")
        or port not in {None, 443}
    ):
        raise ValueError("Use an Azure OpenAI resource URL ending in /openai/v1.")
    if preset.key_required and not key:
        raise ValueError(f"{preset.label} requires an API key in the server configuration.")
    if key and (len(key) > 4096 or not key.isascii() or any(not 33 <= ord(c) <= 126 for c in key)):
        raise ValueError("The configured model API key contains invalid characters.")
    return Endpoint(name, preset.label, preset.protocol, urlunsplit(parsed), remote, key)


def configuration(settings: Settings) -> ModelConfiguration:
    name = settings.llm_provider
    preset = PRESETS.get(name)
    # A custom base URL never receives a key intended for a named hosted endpoint.
    if settings.llm_base_url and name not in {
        "compatible",
        "azure",
        "ollama",
        "lmstudio",
        "llamacpp",
        "vllm",
        "localai",
        "koboldcpp",
    }:
        raise ValueError("Use the compatible provider for a custom hosted API base URL.")
    base = settings.llm_base_url or (
        settings.ollama_host if name == "ollama" else preset.url if preset else ""
    )
    chat = _endpoint(
        name,
        base,
        settings.llm_api_key.get_secret_value(),
        remote_allowed=settings.llm_allow_remote,
    )
    chat_model = settings.llm_chat_model or (settings.ollama_chat_model if name == "ollama" else "")
    # Empty chat IDs are valid on first launch for discovery-based setup.
    if chat_model:
        chat_model = model_id(chat_model)
    utility = (
        settings.llm_utility_model
        or (
            settings.ollama_utility_model
            if name == "ollama" and not settings.llm_chat_model
            else chat_model
        )
        or chat_model
    )
    embed_choice = settings.llm_embed_provider
    embed = None
    if embed_choice == "same":
        if chat.protocol == "anthropic":
            raise ValueError(
                "Claude has no embedding endpoint. Choose Ollama, compatible or disabled."
            )
        if chat.remote and not settings.llm_allow_remote_embeddings:
            raise ValueError("Remote embeddings require LLM_ALLOW_REMOTE_EMBEDDINGS=true.")
        embed = chat
    elif embed_choice == "ollama":
        embed = _endpoint(
            "ollama",
            settings.llm_embed_base_url
            or (chat.base_url if name == "ollama" else settings.ollama_host),
            settings.llm_embed_api_key.get_secret_value() or (chat.key if name == "ollama" else ""),
            remote_allowed=settings.llm_allow_remote_embeddings,
        )
    elif embed_choice == "compatible":
        embed = _endpoint(
            "compatible",
            settings.llm_embed_base_url,
            settings.llm_embed_api_key.get_secret_value(),
            remote_allowed=settings.llm_allow_remote_embeddings,
        )
    elif embed_choice != "disabled":
        raise ValueError("LLM_EMBED_PROVIDER must be ollama, same, compatible or disabled.")
    embed_id = settings.llm_embed_model or (
        settings.ollama_embed_model if embed and embed.name == "ollama" else ""
    )
    return ModelConfiguration(
        chat,
        embed,
        chat_model,
        model_id(utility) if utility else "",
        model_id(embed_id) if embed and embed_id else "",
    )
