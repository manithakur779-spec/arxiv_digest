import logging
from datetime import datetime
from pathlib import Path

from arxiv_digest.settings import PROJECT_ROOT


LOG_DIR = PROJECT_ROOT / "logs"


def configure_logging(log_dir: Path = LOG_DIR) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"log_{timestamp}.txt"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if getattr(handler, "_arxiv_digest_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, mode="x", encoding="utf-8"),
        # logging.StreamHandler(),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._arxiv_digest_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return log_file