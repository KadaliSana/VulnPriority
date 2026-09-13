"""Scanner output to :class:`~vulnpriority.core.models.Scan` (DESIGN 3.1).

Importing this package registers every built-in parser in
:data:`vulnpriority.core.registry.SCANNER_PARSERS`, so
``get_parser("zap")`` works anywhere after ``import vulnpriority.ingest``.

The layer has three jobs and no others: normalise (so the same request described by two
scanners yields one identifier), fingerprint (so applicability has observed versions to
match against), and correlate (so the framework ranks root causes rather than alerts).
It never reaches the network and never consults the clock.
"""

from __future__ import annotations

from vulnpriority.ingest.burp import BurpParser
from vulnpriority.ingest.correlate import (
    FindingCorrelator,
    cluster_sizes,
    clusters,
    dedup_key_for,
)
from vulnpriority.ingest.generic import GenericJsonParser, detect_parser, parse_scan
from vulnpriority.ingest.normalize import (
    ID_PLACEHOLDER,
    EndpointAccumulator,
    canonical_url,
    extract_cves,
    extract_cwe,
    infer_auth_level,
    make_endpoint_id,
    make_finding_id,
    make_scan_id,
    method_is_state_changing,
    severity_from_string,
    template_path,
    templated_path_of,
)
from vulnpriority.ingest.nikto import NiktoParser
from vulnpriority.ingest.nuclei import NucleiParser
from vulnpriority.ingest.wapiti import WapitiParser
from vulnpriority.ingest.tech_fingerprint import (
    fingerprint_library,
    fingerprint_response,
    merge_tech,
)
from vulnpriority.ingest.zap import ZapParser

__all__ = [
    "BurpParser",
    "GenericJsonParser",
    "NiktoParser",
    "NucleiParser",
    "WapitiParser",
    "ZapParser",
    "FindingCorrelator",
    "EndpointAccumulator",
    "ID_PLACEHOLDER",
    "canonical_url",
    "cluster_sizes",
    "clusters",
    "dedup_key_for",
    "detect_parser",
    "extract_cves",
    "extract_cwe",
    "fingerprint_library",
    "fingerprint_response",
    "infer_auth_level",
    "make_endpoint_id",
    "make_finding_id",
    "make_scan_id",
    "merge_tech",
    "method_is_state_changing",
    "parse_scan",
    "severity_from_string",
    "template_path",
    "templated_path_of",
]
