"""Cloud Run startup entrypoint.

Loads baseline data snapshot from GCS (if configured) before launching Streamlit.
"""

import logging
import os
import subprocess
from pathlib import Path

from chatbot.storage import sync_baseline_from_gcs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    data_dir = Path("data")
    sync_baseline_from_gcs(data_dir)

    port = os.getenv("PORT", "8080")
    cmd = [
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    log.info("Starting Streamlit on port %s", port)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
