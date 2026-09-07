"""Centralised loguru configuration.

One place configures all sinks so the rest of the codebase just does
`from loguru import logger` and logs. Called once from the training entry point,
after Hydra has resolved the run directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logging(run_dir: str | Path, debug_to_file: bool = True) -> None:
    """Configure loguru sinks.

    Args:
        run_dir: Directory for the log file (Hydra's per-run output dir).
        debug_to_file: If True, write a DEBUG-level log to run_dir/run.log.
    """
    logger.remove()  # drop the default handler so we control formatting

    logger.add(sys.stderr, level="INFO", format=_FORMAT, colorize=True)

    if debug_to_file:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            run_dir / "run.log",
            level="DEBUG",
            format=_FORMAT,
            colorize=False,
            enqueue=True,  # safe if we ever add multiprocessing dataloaders
        )

    logger.debug("Logging initialised (run_dir={})", run_dir)
