"""Scanner output to :class:`~vulnprio.core.models.Scan` (DESIGN 3.1).

Importing this package registers every built-in parser in
:data:`vulnprio.core.registry.SCANNER_PARSERS`, so
``get_parser("zap")`` works anywhere after ``import vulnprio.ingest``.

The layer has three jobs and no others: normalise (so the same request described by two
scanners yields one identifier), fingerprint (so applicability has observed versions to
match against), and correlate (so the framework ranks root causes rather than alerts).
It never reaches the network and never consults the clock.
"""

from __future__ import annotations

from vulnprio.ingest.burp import BurpParser
from vulnprio.ingest.correlate import (
    FindingCorrelator,
    cluster_sizes,
    clusters,
    dedup_key_for,
)
from vulnprio.ingest.generic import GenericJsonParser, detect_parser, parse_scan
from vulnprio.ingest.normalize import (
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
from vulnprio.ingest.nikto import NiktoParser
from vulnprio.ingest.nuclei import NucleiParser
from vulnprio.ingest.wapiti import WapitiParser
from vulnprio.ingest.tech_fingerprint import (
    fingerprint_library,
    fingerprint_response,
    merge_tech,
)
from vulnprio.ingest.zap import ZapParser

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
