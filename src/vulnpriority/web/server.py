"""A development server for the exported site.

The site works from ``file://``; this exists only so that ``--serve`` can hand someone a URL.
It binds to localhost, serves one directory, and logs nothing to the console by default.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

__all__ = ["serve_site"]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover - noise only
        return


def serve_site(
    directory: str | Path,
    port: int = 8000,
    host: str = "127.0.0.1",
    open_browser: bool = False,
    block: bool = True,
) -> tuple[socketserver.TCPServer, str]:
    """Serve ``directory`` over HTTP and return the server and its URL."""
    directory = Path(directory)
    if not (directory / "index.html").exists():
        raise FileNotFoundError(f"no exported site at {directory} (run 'vulnpriority web' first)")

    handler = functools.partial(_QuietHandler, directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer((host, port), handler)
    url = f"http://{host}:{httpd.server_address[1]}/index.html"

    if open_browser:  # pragma: no cover - depends on the desktop
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    if block:  # pragma: no cover - blocks until interrupted
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
    else:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, url
