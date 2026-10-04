"""Entry point:  ``python run.py``

Starts the Gradio UI. All configuration is read from environment variables
prefixed with ``APACHE_`` -- see ``app/config.py`` for the full list.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apache - a local, voice-first AI assistant."
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--port", type=int, default=7860, help="port to listen on")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )
    parser.add_argument("--share", action="store_true", help="create a public link")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report on the environment and exit without starting the server",
    )
    args = parser.parse_args(argv)

    if args.check:
        from app.check import environment_report

        return environment_report()

    # Imported lazily so `--check` works even when the ML stack is unavailable.
    from app.ui import build_and_launch

    build_and_launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
