import os

from manual_graphrag.ui import build_app


def launch_settings() -> dict[str, object]:
    return {
        "server_name": "0.0.0.0",
        "server_port": 8080,
        "share": False,
        "inbrowser": os.getenv("GRADIO_INBROWSER", "true").lower()
        in {"1", "true", "yes"},
    }


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(**launch_settings())
