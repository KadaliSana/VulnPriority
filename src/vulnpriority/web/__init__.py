"""Static website that illustrates a pipeline run.

``vulnpriority web`` exports a self-contained page: open ``index.html`` from disk, no server and
no network required. ``serve_site`` is a convenience wrapper around the standard library.
"""

from vulnpriority.web.exporter import ASSET_DIR, build_dashboard, export_site, write_payload
from vulnpriority.web.schema import DASHBOARD_SCHEMA_VERSION, DashboardData

__all__ = [
    "ASSET_DIR",
    "build_dashboard",
    "export_site",
    "write_payload",
    "DashboardData",
    "DASHBOARD_SCHEMA_VERSION",
]
