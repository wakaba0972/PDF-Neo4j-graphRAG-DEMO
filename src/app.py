import os
from pathlib import Path

from manual_graphrag.ui import build_app

# .env, config/ and data/ are all resolved relative to the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def launch_settings() -> dict[str, object]:
    return {
        "server_name": "0.0.0.0",
        "server_port": 8080,
        "share": False,
        "inbrowser": os.getenv("GRADIO_INBROWSER", "true").lower()
        in {"1", "true", "yes"},
        # Gradio's localhost HEAD probe can time out while the large API schema
        # is generated on the first page request. The startup-events GET check
        # still runs, so skip only that redundant frontend reachability probe.
        "_frontend": False,
    }


if __name__ == "__main__":
    # Launching from another directory (e.g. `python PDF-.../src/app.py`) would
    # otherwise silently miss .env and fall back to empty API keys.
    os.chdir(PROJECT_ROOT)
    build_app().queue(default_concurrency_limit=1).launch(**launch_settings())
