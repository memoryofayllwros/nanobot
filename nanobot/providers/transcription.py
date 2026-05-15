"""Voice transcription providers (Groq, OpenAI Whisper, OpenRouter STT)."""

import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

_OPENROUTER_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "nanobot",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}

# Up to 3 retries (4 attempts total) with exponential backoff on transient
# failures. Whisper endpoints occasionally return 502/503 under load, and
# mobile-network transcription callers hit sporadic connect/read errors.
# Without this, a voice message silently becomes the empty string.
_MAX_RETRIES = 3
_BACKOFF_S = (1.0, 2.0, 4.0)
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


async def _post_transcription_with_retry(
    url: str,
    *,
    api_key: str | None,
    path: Path,
    model: str,
    provider_label: str,
    language: str | None = None,
) -> str:
    """POST an audio file for transcription, retrying on transient errors.

    Retries on connect/read/timeout failures and on 408/429/5xx responses.
    Other errors (including 4xx such as 401/403) return "" immediately — the
    caller's config is wrong and retrying only wastes quota.

    When ``language`` is provided, it is forwarded as the ``language``
    multipart field on every attempt (the dict is rebuilt per attempt so the
    same field is present on retries).
    """
    try:
        data = path.read_bytes()
    except OSError as e:
        logger.exception("{} transcription error: cannot read audio file: {}", provider_label, e)
        return ""
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient() as client:
        for attempt in range(_MAX_RETRIES + 1):
            files = {
                "file": (path.name, data),
                "model": (None, model),
            }
            if language:
                files["language"] = (None, language)
            try:
                response = await client.post(url, headers=headers, files=files, timeout=60.0)
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "{} transcription transient error (attempt {}/{}): {}",
                        provider_label,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        e,
                    )
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                logger.exception(
                    "{} transcription error after {} attempts: {}",
                    provider_label,
                    _MAX_RETRIES + 1,
                    e,
                )
                return ""
            except Exception as e:
                logger.exception("{} transcription error: {}", provider_label, e)
                return ""

            if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                logger.warning(
                    "{} transcription transient HTTP {} (attempt {}/{})",
                    provider_label,
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                )
                await asyncio.sleep(_BACKOFF_S[attempt])
                continue

            try:
                response.raise_for_status()
            except Exception as e:
                logger.exception("{} transcription error: {}", provider_label, e)
                return ""

            try:
                payload = response.json()
            except Exception as e:
                logger.exception(
                    "{} transcription error: malformed response body: {}",
                    provider_label,
                    e,
                )
                return ""
            if not isinstance(payload, dict):
                logger.error(
                    "{} transcription error: unexpected response shape: {!r}",
                    provider_label,
                    type(payload).__name__,
                )
                return ""
            return payload.get("text", "")


def _openrouter_audio_format(path: Path) -> str:
    """Map file to OpenRouter ``input_audio.format`` (docs: wav, mp3, ogg, …)."""
    ext = path.suffix.lower()
    by_ext = {
        ".wav": "wav",
        ".wave": "wav",
        ".mp3": "mp3",
        ".mpeg": "mp3",
        ".mpga": "mp3",
        ".ogg": "ogg",
        ".opus": "ogg",
        ".oga": "ogg",
        ".m4a": "m4a",
        ".mp4": "m4a",
        ".webm": "webm",
        ".aac": "aac",
        ".flac": "flac",
    }
    if ext in by_ext:
        return by_ext[ext]
    mime = mimetypes.guess_type(str(path))[0] or ""
    if "wav" in mime:
        return "wav"
    if "mpeg" in mime or "mp3" in mime:
        return "mp3"
    if "ogg" in mime:
        return "ogg"
    if "mp4" in mime or "m4a" in mime:
        return "m4a"
    if "webm" in mime:
        return "webm"
    if "aac" in mime:
        return "aac"
    if "flac" in mime:
        return "flac"
    return "mp3"


async def _post_openrouter_stt_with_retry(
    *,
    api_url: str,
    api_key: str,
    path: Path,
    model: str,
    language: str | None,
    provider_label: str,
) -> str:
    try:
        raw = path.read_bytes()
    except OSError as e:
        logger.exception("{} transcription error: cannot read audio file: {}", provider_label, e)
        return ""
    fmt = _openrouter_audio_format(path)
    payload: dict[str, object] = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(raw).decode("ascii"),
            "format": fmt,
        },
    }
    if language:
        payload["language"] = language

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **_OPENROUTER_ATTRIBUTION_HEADERS,
    }

    async with httpx.AsyncClient() as client:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.post(api_url, headers=headers, json=payload, timeout=120.0)
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "{} transcription transient error (attempt {}/{}): {}",
                        provider_label,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        e,
                    )
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                logger.exception(
                    "{} transcription error after {} attempts: {}",
                    provider_label,
                    _MAX_RETRIES + 1,
                    e,
                )
                return ""
            except Exception as e:
                logger.exception("{} transcription error: {}", provider_label, e)
                return ""

            if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                logger.warning(
                    "{} transcription transient HTTP {} (attempt {}/{})",
                    provider_label,
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                )
                await asyncio.sleep(_BACKOFF_S[attempt])
                continue

            if not response.is_success:
                logger.warning(
                    "{} STT HTTP {} — {!r}",
                    provider_label,
                    response.status_code,
                    (response.text or "")[:800],
                )
                return ""

            try:
                body = response.json()
            except Exception as e:
                logger.exception(
                    "{} transcription error: malformed response body: {}",
                    provider_label,
                    e,
                )
                return ""
            if not isinstance(body, dict):
                logger.error(
                    "{} transcription error: unexpected response shape: {!r}",
                    provider_label,
                    type(body).__name__,
                )
                return ""
            err = body.get("error")
            if err:
                logger.warning("{} STT API error in body: {!r}", provider_label, err)
                return ""
            return str(body.get("text", "") or "").strip()


async def _openrouter_chat_input_audio_transcribe(
    *,
    api_base: str,
    api_key: str,
    path: Path,
    model: str,
    language: str | None,
    provider_label: str,
) -> str:
    """Transcribe via OpenRouter ``/chat/completions`` with ``input_audio`` (Gemini, etc.)."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        logger.exception("{} chat-audio error: cannot read file: {}", provider_label, e)
        return ""
    fmt = _openrouter_audio_format(path)
    b64 = base64.b64encode(raw).decode("ascii")
    lang_hint = f" The audio is likely in {language!r}." if language else ""
    prompt = (
        "Transcribe the speech in this audio verbatim. Output only the spoken words; "
        "use the same language as the audio. No preamble, labels, or explanation."
        + lang_hint
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": fmt},
                    },
                ],
            }
        ],
        "stream": False,
    }
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **_OPENROUTER_ATTRIBUTION_HEADERS,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=120.0)
        except Exception as e:
            logger.exception("{} chat-audio request failed: {}", provider_label, e)
            return ""

        if not response.is_success:
            logger.warning(
                "{} chat-audio HTTP {} — {!r}",
                provider_label,
                response.status_code,
                (response.text or "")[:800],
            )
            return ""
        try:
            body = response.json()
        except Exception as e:
            logger.exception("{} chat-audio bad JSON: {}", provider_label, e)
            return ""

    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if err:
        logger.warning("{} chat-audio API error: {!r}", provider_label, err)
        return ""

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()
    return ""
    """Voice transcription provider using OpenAI's Whisper API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = (
            api_base
            or os.environ.get("OPENAI_TRANSCRIPTION_BASE_URL")
            or "https://api.openai.com/v1/audio/transcriptions"
        )
        self.language = language or None

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("OpenAI API key not configured for transcription")
            return ""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        return await _post_transcription_with_retry(
            self.api_url,
            api_key=self.api_key,
            path=path,
            model="whisper-1",
            provider_label="OpenAI",
            language=self.language,
        )


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = (
            api_base
            or os.environ.get("GROQ_BASE_URL")
            or "https://api.groq.com/openai/v1/audio/transcriptions"
        )
        self.language = language or None

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        return await _post_transcription_with_retry(
            self.api_url,
            api_key=self.api_key,
            path=path,
            model="whisper-large-v3",
            provider_label="Groq",
            language=self.language,
        )


class OpenRouterTranscriptionProvider:
    """Speech-to-text via OpenRouter ``/audio/transcriptions`` (e.g. Gemini)."""

    DEFAULT_MODEL = "google/gemini-3.1-flash-lite"

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._api_base = (
            api_base or os.environ.get("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.api_url = f"{self._api_base}/audio/transcriptions"
        self.language = language or None
        self.model = (model or "").strip() or self.DEFAULT_MODEL

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("OpenRouter API key not configured for transcription")
            return ""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        via_stt = await _post_openrouter_stt_with_retry(
            api_url=self.api_url,
            api_key=self.api_key,
            path=path,
            model=self.model,
            language=self.language,
            provider_label="OpenRouter",
        )
        if via_stt.strip():
            return via_stt

        logger.info(
            "OpenRouter dedicated STT returned empty; trying chat/completions input_audio model={}",
            self.model,
        )
        return await _openrouter_chat_input_audio_transcribe(
            api_base=self._api_base,
            api_key=self.api_key,
            path=path,
            model=self.model,
            language=self.language,
            provider_label="OpenRouter",
        )
