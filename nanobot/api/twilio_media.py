"""Download Twilio-hosted media referenced as ``[media:<idx>:<url>]`` lines."""

from __future__ import annotations

import io
import mimetypes
import os
import re
import uuid
import zipfile
from pathlib import Path
from urllib.parse import unquote

import httpx
from loguru import logger

from nanobot.utils.helpers import safe_filename
from nanobot.utils.media_decode import MAX_FILE_SIZE

# Inbound WhatsApp / SMS convention from JoJo bridge: one line per attachment.
_MEDIA_LINE_RE = re.compile(
    r"(?m)^\[media:(?P<idx>\d+):(?P<url>https?://[^\]\s]+)\]\s*$",
)
_MEDIA_CONTENT_TYPE_LINE_RE = re.compile(
    r"^\[media_content_type:(?P<idx>\d+):(?P<content_type>[^\]]+)\]\s*$",
)

_TWILIO_MEDIA_HOST_MARKERS = ("api.twilio.com", "messaging.twilio.com")


def twilio_rest_credentials() -> tuple[str, str] | None:
    sid = (os.getenv("ACCOUNT_SID") or os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    token = (os.getenv("AUTH_TOKEN") or os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    if not sid or not token:
        return None
    return sid, token


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split(";"):
        part = part.strip()
        if part.lower().startswith("filename*="):
            try:
                _, enc_name = part.split("=", 1)
                enc_name = enc_name.strip().strip('"')
                if "''" in enc_name:
                    enc_name = enc_name.split("''", 1)[1]
                return unquote(enc_name) or None
            except ValueError:
                continue
        if part.lower().startswith("filename="):
            name = part.split("=", 1)[1].strip().strip('"')
            return name or None
    return None


def _extension_from_url_path(url: str) -> str | None:
    path = url.split("?", 1)[0]
    suffix = Path(path).suffix.lower()
    if suffix in {
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".txt",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".mp3",
        ".m4a",
        ".ogg",
        ".opus",
        ".amr",
        ".wav",
        ".webm",
    }:
        return suffix
    return None


def _guess_extension(content_type: str | None, url: str) -> str:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    ext = _extension_from_url_path(url)
    if ext:
        return ext
    if ct:
        guessed = mimetypes.guess_extension(ct)
        if guessed:
            return guessed
        if ct == "application/pdf":
            return ".pdf"
        if ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return ".docx"
        if ct == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return ".xlsx"
        if ct == "audio/ogg":
            return ".ogg"
        if ct in {"audio/opus", "audio/ogg; codecs=opus"}:
            return ".ogg"
        if ct == "audio/amr":
            return ".amr"
        if ct == "audio/mpeg":
            return ".mp3"
        if ct in {"audio/mp4", "audio/aac"}:
            return ".m4a"
        if ct == "audio/webm":
            return ".webm"
        if ct in {"audio/wav", "audio/x-wav"}:
            return ".wav"
    return ".bin"


def _guess_extension_from_bytes(raw: bytes) -> str | None:
    if raw.startswith(b"%PDF"):
        return ".pdf"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return ".gif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp"
    if raw.startswith(b"OggS"):
        return ".ogg"
    if raw.startswith(b"ID3") or raw[:2] == b"\xff\xfb":
        return ".mp3"
    if raw.startswith(b"#!AMR"):
        return ".amr"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
        return ".wav"
    if raw.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return ".zip"
        if "word/document.xml" in names:
            return ".docx"
        if "xl/workbook.xml" in names:
            return ".xlsx"
        if "ppt/presentation.xml" in names:
            return ".pptx"
        return ".zip"
    return None


def _is_twilio_media_url(url: str) -> bool:
    u = url.lower()
    return any(h in u for h in _TWILIO_MEDIA_HOST_MARKERS)


async def _download_one(
    url: str,
    media_dir: Path,
    creds: tuple[str, str],
    content_type_hint: str | None = None,
) -> str | None:
    max_bytes = int(os.getenv("NANOBOT_TWILIO_MEDIA_MAX_BYTES") or str(MAX_FILE_SIZE))
    try:
        async with httpx.AsyncClient(
            auth=creds,
            timeout=httpx.Timeout(120.0),
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    peek = (await resp.aread())[:500]
                    logger.warning(
                        "Twilio media GET {} -> {}: {}",
                        url[:120],
                        resp.status_code,
                        peek,
                    )
                    return None
                cd = resp.headers.get("content-disposition")
                ct = content_type_hint or resp.headers.get("content-type")
                fname_hint = _filename_from_content_disposition(cd)
                size = 0
                chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        logger.warning(
                            "Twilio media exceeds max_bytes={} url={}",
                            max_bytes,
                            url[:120],
                        )
                        return None
                    chunks.append(chunk)
                raw = b"".join(chunks)
                ext = _guess_extension_from_bytes(raw) or _guess_extension(ct, url)
                if fname_hint:
                    base = f"{uuid.uuid4().hex[:12]}_{safe_filename(fname_hint)}"
                    if Path(base).suffix.lower() in {"", ".bin"} and ext != ".bin":
                        base = f"{Path(base).stem}{ext}"
                else:
                    base = f"{uuid.uuid4().hex[:12]}{ext}"
                dest = media_dir / safe_filename(base)
                dest.write_bytes(raw)
                logger.info(
                    "Saved Twilio media -> {} ({} bytes, ct={})",
                    dest.name,
                    size,
                    (ct or "")[:80],
                )
                return str(dest)
    except Exception as e:
        logger.warning("Twilio media download failed url={}: {}", url[:120], e)
        return None


async def ingest_twilio_media_lines(text: str, media_dir: Path) -> tuple[str, list[str]]:
    """Replace ``[media:*:https://api.twilio.com/...]`` lines with saved files.

    Returns ``(updated_text, local_paths)``. Non-Twilio ``[media:...]`` lines are
    left unchanged. Download failures keep the original line and log a warning.
    """
    creds = twilio_rest_credentials()
    if not creds:
        return text, []

    media_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    new_lines: list[str] = []
    content_type_hints: dict[str, str] = {}
    for line in text.splitlines():
        type_match = _MEDIA_CONTENT_TYPE_LINE_RE.match(line)
        if type_match:
            content_type_hints[type_match.group("idx")] = type_match.group("content_type").strip()
            continue

        m = _MEDIA_LINE_RE.match(line)
        if not m:
            new_lines.append(line)
            continue
        url = m.group("url").strip()
        if not _is_twilio_media_url(url):
            new_lines.append(line)
            continue
        path = await _download_one(url, media_dir, creds, content_type_hints.get(m.group("idx")))
        if path:
            saved_paths.append(path)
        else:
            new_lines.append(line)

    new_text = "\n".join(new_lines).strip()
    return new_text, saved_paths
