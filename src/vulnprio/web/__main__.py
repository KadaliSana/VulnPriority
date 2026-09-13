"""``python -m vulnprio.web <run_dir> [--out DIR] [--serve]``.

A thin entry point so the site can be exported without the full CLI, which matters when
only part of the pipeline has been run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vulnprio.web", description="Export the results website.")
    parser.add_argument("run_dir", nargs="?", help="run directory to read artifacts from")
    parser.add_argument("--demo", action="store_true", help="render a worked example instead of a real run")
    parser.add_argument("--out", default="site", help="output directory (default: site)")
    parser.add_argument("--serve", action="store_true", help="serve the exported site")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    from vulnprio.web.exporter import build_dashboard, export_site
    from vulnprio.web.loader import load_run

    if args.demo:
        from vulnprio.web.demo import demo_dashboard

        data = demo_dashboard()
    elif args.run_dir:
        data = load_run(Path(args.run_dir))
    else:
        parser.error("give a run directory, or --demo to render the worked example")
    out = export_site(data, args.out)
    print(f"site written to {out.resolve()}")
    if args.serve:
        from vulnprio.web.server import serve_site

        _, url = serve_site(out, port=args.port, open_browser=True)
        print(f"serving at {url}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
