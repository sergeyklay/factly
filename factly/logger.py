"""Logging utilities for factly."""

from __future__ import annotations

import logging
import sys


class DebugFilter(logging.Filter):
    """Extended standard logging Filter to filer only DEBUG messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Only messages with record level DEBUG can pass
        for messages with another level an extra handler is used.

        :param tuple record: logging message record
        :returns: True|False
        :rtype: bool
        """
        return record.levelno == logging.DEBUG


class InfoFilter(logging.Filter):
    """Extended standard logging Filter to filer only INFO messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Only messages with record level INFO can pass
        for messages with another level an extra handler is used.

        :param tuple record: logging message record
        :returns: True|False
        :rtype: bool
        """
        return record.levelno == logging.INFO


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure module-specific logging without affecting third-party loggers.

    Args:
        verbose: Whether to show detailed information
    """
    root = logging.getLogger("factly")
    root.setLevel(logging.ERROR if quiet else logging.DEBUG)

    formatter = logging.Formatter("%(message)s")

    specs = (
        {"on": verbose, "filter": DebugFilter()},
        {"on": not quiet, "level": logging.INFO, "filter": InfoFilter()},
        {"on": True, "stream": sys.stderr, "level": logging.ERROR},
    )

    for spec in specs:
        if spec["on"]:
            handler = logging.StreamHandler(spec.get("stream", sys.stdout))
            handler.setLevel(spec.get("level", logging.DEBUG))
            handler.setFormatter(formatter)

            if spec.get("filter"):
                handler.addFilter(spec["filter"])

            root.addHandler(handler)

    return root
