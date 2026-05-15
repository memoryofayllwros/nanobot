"""Built-in WhatsApp channel registration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.channels.base import BaseChannel

_BUILTIN_CHANNELS: tuple[str, ...] = ("whatsapp",)


def discover_channel_names() -> list[str]:
    """Return built-in channel module names (WhatsApp only in this fork)."""
    return list(_BUILTIN_CHANNELS)


def load_channel_class(module_name: str) -> type[BaseChannel]:
    """Import *module_name* and return the first BaseChannel subclass found."""
    from nanobot.channels.base import BaseChannel as _Base

    mod = importlib.import_module(f"nanobot.channels.{module_name}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, _Base) and obj is not _Base:
            return obj
    raise ImportError(f"No BaseChannel subclass in nanobot.channels.{module_name}")


def discover_all() -> dict[str, type[BaseChannel]]:
    """Return the WhatsApp channel class."""
    builtin: dict[str, type[BaseChannel]] = {}
    for modname in discover_channel_names():
        try:
            builtin[modname] = load_channel_class(modname)
        except ImportError as e:
            logger.debug("Skipping built-in channel '{}': {}", modname, e)
    return builtin
