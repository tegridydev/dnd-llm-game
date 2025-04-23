import logging
from typing import Any, Dict, List
from requests.exceptions import ConnectionError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from ollama import (
    chat as ollama_chat,
    generate as ollama_generate,
    embed as ollama_embed,
    list as ollama_list_models,
    show as ollama_show,
    pull as ollama_pull,
    push as ollama_push,
    create as ollama_create,
    copy as ollama_copy,
    delete as ollama_delete,
    ps as ollama_ps,
)
from ollama._types import ResponseError
from core.settings import settings

logger = logging.getLogger(__name__)

_retry = retry(
    retry=retry_if_exception_type((ResponseError, ConnectionError)),
    wait=wait_exponential(min=1, max=5),
    stop=stop_after_attempt(3),
    reraise=True,
)

class OllamaClient:
    """
    Thin wrapper around Ollama’s HTTP API—separates chat vs generate.
    """

    @_retry
    def chat(self, messages: List[Dict[str, Any]], stream: bool=False) -> Any:
        try:
            return ollama_chat(model=settings.ollama_model, messages=messages, stream=stream)
        except ResponseError as e:
            if e.status_code == 404:
                logger.warning("Model not found, pulling...")
                ollama_pull(model=settings.ollama_model)
                return ollama_chat(model=settings.ollama_model, messages=messages, stream=stream)
            raise

    @_retry
    def generate(
        self,
        prompt: str,
        suffix: str="",
        max_tokens: int=150,
        temperature: float=0.8,
        stream: bool=False,
    ) -> Any:
        opts = {"temperature": temperature, "num_predict": max_tokens}
        try:
            return ollama_generate(
                model=settings.ollama_model,
                prompt=prompt,
                suffix=suffix,
                options=opts,
                stream=stream,
            )
        except ResponseError as e:
            if e.status_code == 404:
                logger.warning("Model not found, pulling...")
                ollama_pull(model=settings.ollama_model)
                return ollama_generate(
                    model=settings.ollama_model,
                    prompt=prompt,
                    suffix=suffix,
                    options=opts,
                    stream=stream,
                )
            raise

    def embed(self, inputs: List[str]) -> Any:
        return ollama_embed(model=settings.ollama_model, input=inputs)

    def list_models(self) -> Any:
        return ollama_list_models()

    def show(self, model: str) -> Any:
        return ollama_show(model)

    def pull(self, model: str) -> Any:
        return ollama_pull(model, stream=True)

    def push(self, model: str) -> Any:
        return ollama_push(model, insecure=True, stream=True)

    def create(self, **kwargs: Any) -> Any:
        return ollama_create(**kwargs)

    def copy(self, src: str, dst: str) -> Any:
        return ollama_copy(src, dst)

    def delete(self, model: str) -> Any:
        return ollama_delete(model)

    def ps(self) -> Any:
        return ollama_ps()

ollama_client = OllamaClient()
