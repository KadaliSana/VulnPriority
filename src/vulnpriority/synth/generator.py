"""``SyntheticDataset.generate``: the whole generated world, on disk (DESIGN.md 3.11).

What this produces, under ``data/synthetic/<dataset hash>/``:

``scans/<scan_id>.json``
    Canonical :class:`~vulnpriority.core.models.Scan` documents - the same format
    :class:`~vulnpriority.ingest.generic.GenericJsonParser` reads, so the pipeline ingests
    synthetic data through exactly the code path it uses for a real ZAP report.
``feeds/``
    A complete fixture feed directory, usable as ``FeedsConfig.fixture_dir`` unmodified.
``oracle.json`` / ``world.json``
    The latent ground truth, kept beside the data but never read by anything upstream of
    ``eval/``: the framework's own stages are handed the scans and the feeds only.
``dataset.json``
    The dataset manifest: hash, configuration, counts and the scan index.

Three properties the evaluation protocol depends on:

* **Determinism.** Every draw comes from a seeded generator derived by SHA-256 from the run
  seed and a label, so the same configuration produces byte-identical files, and adding an
  application does not perturb the ones before it.
* **Time.** Applications are staggered and scanned repeatedly, so scan dates interleave
  across applications and a time-ordered split with a 30-day gap has something to cut.
* **Persistence.** A defect observed in scan 1 is the *same defect* in scans 2 and 3 - same
  ``defect_key``, one exploitation event - which is what makes the longitudinal exposure
  simulation measure remediation timing rather than resampling noise.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from vulnpriority.core.config import PROJECT_ROOT, PipelineConfig, SyntheticConfig
from vulnpriority.core.enums import (
    EndpointFunction,
    PrivilegeLevel,
    Provenance,
    ScannerSeverity,
)
from vulnpriority.core.hashing import dataset_hash, stable_id
from vulnpriority.core.models import (
    AttackerModel,
    Endpoint,
    Finding,
    Scan,
    TechComponent,
    UntrustedText,
)
from vulnpriority.ingest.correlate import FindingCorrelator
from vulnpriority.ingest.normalize import make_finding_id, make_scan_id
from vulnpriority.synth.feeds import write_feed_fixtures
from vulnpriority.synth.oracle import ExploitationOracle, LatentFinding, simulate_exploitation
from vulnpriority.synth.pages import generate_reference_pages
from vulnpriority.synth.topology import (
    AppSpec,
    derive_seed,
    generate_app_specs,
    generate_endpoints,
    intended_function,
)
from vulnpriority.synth.world import VULN_TEMPLATES, LatentWorld, build_world, clamp01

__all__ = [
    "DEFAULT_SYNTHETIC_ROOT",
    "LATENT_FUNCTION_VALUE",
    "SyntheticDataset",
    "synthetic_config_of",
]

#: Where a generated world lands when the caller does not say otherwise.
DEFAULT_SYNTHETIC_ROOT: Path = PROJECT_ROOT / "data" / "synthetic"

#: Latent worth of an endpoint by its *intended* function. Read only by the oracle. The
#: framework infers its own criticality from structure and never sees these numbers, which
#: is precisely what makes "did Component A find the valuable assets?" a real question.
LATENT_FUNCTION_VALUE: dict[EndpointFunction, float] = {
    EndpointFunction.PAYMENT: 0.95,
    EndpointFunction.ADMIN: 0.92,
    EndpointFunction.PII_DATA: 0.88,
    EndpointFunction.AUTH: 0.74,
    EndpointFunction.FILE_IO: 0.66,
    EndpointFunction.API_DATA: 0.58,
    EndpointFunction.SEARCH: 0.38,
    EndpointFunction.STATIC_CONTENT: 0.14,
    EndpointFunction.UNKNOWN: 0.35,
}

#: Scanner severity ladder, for the one-step mis-grading real scanners do.
_SEVERITY_ORDER: tuple[ScannerSeverity, ...] = (
    ScannerSeverity.INFO,
    ScannerSeverity.LOW,
    ScannerSeverity.MEDIUM,
    ScannerSeverity.HIGH,
    ScannerSeverity.CRITICAL,
)

_EVIDENCE_SNIPPETS: dict[int, str] = {
    89: "payload ' OR '1'='1 -- returned HTTP 200 with 1284 rows",
    79: "payload <script>alert(1)</script> was reflected unencoded in the response body",
    78: "payload ;id returned uid=33(www-data) gid=33(www-data)",
    22: "payload ../../../../etc/passwd returned root:x:0:0:root:/root:/bin/bash",
    502: "a serialised object header (rO0AB) was accepted and reconstructed",
    918: "the server issued an outbound request to http://169.254.169.254/latest/meta-data/",
    287: "a request with an empty signature header returned HTTP 200 and a session cookie",
    862: "an unprivileged session received HTTP 200 from an administrative function",
    639: "changing the identifier returned a record belonging to another account",
    434: "a file named shell.php was accepted and served back with content type text/x-php",
    352: "a cross-origin POST without a token changed server state and returned HTTP 302",
    611: "an external entity declaration was resolved and its contents returned",
    601: "the Location header echoed the attacker-supplied host verbatim",
    798: "the response body contained the string password=Adm1nS3cret",
    200: "the Server header disclosed the exact product build and patch level",
    94: "payload {{7*7}} evaluated to 49 in the rendered response",
}


def synthetic_config_of(config: SyntheticConfig | PipelineConfig | None) -> SyntheticConfig:
    """Accept the whole pipeline config, just its synthetic section, or nothing."""
    if config is None:
        return SyntheticConfig()
    if isinstance(config, PipelineConfig):
        return config.synthetic
    if isinstance(config, SyntheticConfig):
        return config
    raise TypeError(f"expected SyntheticConfig or PipelineConfig, got {type(config).__name__}")


def _default_attacker() -> AttackerModel:
    """The attacker whose behaviour the oracle simulates.

    The opportunistic preset is the honest default for a synthetic world: it is the
    adversary most web applications actually face, and it is the one whose target
    preferences are least likely to flatter a chain-aware ranker.
    """
    try:
        from vulnpriority.attacker.presets import load_preset

        return load_preset("opportunistic")
    except Exception:
        return AttackerModel(name="opportunistic")


# ---------------------------------------------------------------------------
# Defect roster
# ---------------------------------------------------------------------------


class _Defect(BaseModel):
    """One underlying defect of an application, persisting across scans."""

    model_config = ConfigDict(frozen=True)

    defect_key: str
    endpoint_id: str
    cwe_id: int
    plugin_id: str
    name: str
    title: str
    param: str
    cve_id: str | None
    severity: ScannerSeverity
    confidence: float
    first_scan: int
    last_scan: int
    function: EndpointFunction
    latent_exposure: float
    latent_asset_value: float
    latent_chain_depth: int
    latent_applies: bool
    privilege_gained: PrivilegeLevel
    affected_product: str | None = None


def _latent_exposure(endpoint: Endpoint) -> float:
    if not endpoint.internet_facing:
        return 0.15
    if endpoint.auth_required == PrivilegeLevel.NONE:
        return 1.0
    if endpoint.auth_required == PrivilegeLevel.USER:
        return 0.55
    return 0.3


def _chain_depth(endpoint: Endpoint) -> int:
    return int(endpoint.auth_required)


def _noisy_severity(base: ScannerSeverity, rng: Random) -> ScannerSeverity:
    """Scanner severity with the one-step mis-grading real tools exhibit.

    A scanner's severity is a noisy instrument, not a measurement, and a synthetic world in
    which it were exact would make the scanner-severity baseline artificially strong and the
    whole framework artificially unnecessary.
    """
    index = _SEVERITY_ORDER.index(base)
    roll = rng.random()
    if roll < 0.14 and index > 0:
        return _SEVERITY_ORDER[index - 1]
    if roll > 0.88 and index < len(_SEVERITY_ORDER) - 1:
        return _SEVERITY_ORDER[index + 1]
    return base


def _build_roster(
    app: AppSpec,
    endpoints: Sequence[Endpoint],
    world: LatentWorld,
    config: SyntheticConfig,
    rng: Random,
) -> tuple[_Defect, ...]:
    """The defects one application carries, with the scans each is visible in."""
    low, high = int(config.findings_per_app[0]), int(config.findings_per_app[1])
    n_defects = rng.randint(min(low, high), max(low, high))
    scans = max(1, int(config.scans_per_app))

    stack_products = {component.product for component in app.tech_stack}
    by_function: dict[EndpointFunction, list[Endpoint]] = {}
    for endpoint in endpoints:
        by_function.setdefault(intended_function(endpoint.path), []).append(endpoint)

    vulns_by_cwe: dict[int, list] = {}
    for vuln in world.vulns:
        vulns_by_cwe.setdefault(vuln.cwe_id, []).append(vuln)

    roster: list[_Defect] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for _ in range(n_defects):
        template = VULN_TEMPLATES[rng.randrange(len(VULN_TEMPLATES))]
        candidates = [
            endpoint
            for function in template.functions
            for endpoint in by_function.get(function, ())
        ] or list(endpoints)
        endpoint = candidates[rng.randrange(len(candidates))]
        param = endpoint.parameters[rng.randrange(len(endpoint.parameters))] if endpoint.parameters else ""

        cve_id: str | None = None
        version_applies = True
        affected_product: str | None = None
        pool = vulns_by_cwe.get(template.cwe_id, [])
        if pool and rng.random() < template.carries_cve:
            # Prefer a CVE about something this application actually runs, so version
            # matching has a decision to make rather than defaulting to UNKNOWN.
            local = [vuln for vuln in pool if vuln.product in stack_products]
            chosen = (local or pool)[rng.randrange(len(local or pool))]
            cve_id = chosen.cve_id
            version_applies = bool(chosen.version_applies and chosen.product in stack_products)
            affected_product = chosen.product if chosen.product in stack_products else None
            applies = version_applies and rng.random() < chosen.true_applicability
        else:
            applies = rng.random() < 0.86

        # ``finding_id`` is derived from (scan, endpoint, plugin id, parameter), so two
        # defects sharing that tuple would collide into one finding and silently drop a
        # row. The roster is therefore keyed on exactly that tuple, not on the defect
        # identity, and a collision skips rather than overwrites.
        identity = (endpoint.endpoint_id, template.plugin_id, param)
        if identity in seen_keys:
            continue
        seen_keys.add(identity)
        defect_key = stable_id(
            "defect", app.app_id, endpoint.endpoint_id, str(template.cwe_id), cve_id or "", param
        )

        first_scan = 0 if rng.random() < 0.7 else rng.randrange(scans)
        last_scan = scans - 1 if rng.random() < 0.78 else rng.randint(first_scan, scans - 1)
        function = intended_function(endpoint.path)
        roster.append(
            _Defect(
                defect_key=defect_key,
                endpoint_id=endpoint.endpoint_id,
                cwe_id=template.cwe_id,
                plugin_id=template.plugin_id,
                name=template.name,
                title=template.title.format(param=param or "request body"),
                param=param,
                cve_id=cve_id,
                severity=_noisy_severity(template.severity, rng),
                confidence=round(clamp01(rng.betavariate(5.0, 2.0)), 3),
                first_scan=first_scan,
                last_scan=last_scan,
                function=function,
                latent_exposure=_latent_exposure(endpoint),
                latent_asset_value=round(
                    clamp01(
                        LATENT_FUNCTION_VALUE.get(function, 0.35) + rng.normalvariate(0.0, 0.07)
                    ),
                    4,
                ),
                latent_chain_depth=_chain_depth(endpoint),
                latent_applies=applies,
                privilege_gained=template.privilege_gained,
                affected_product=affected_product,
            )
        )
    return tuple(roster)


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


class SyntheticDataset(BaseModel):
    """A generated world: scans, feed fixtures, reference pages and the latent oracle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SyntheticConfig = Field(default_factory=SyntheticConfig)
    dataset_hash: str = ""
    scans: tuple[Scan, ...] = ()
    world: LatentWorld
    oracle: ExploitationOracle
    latent_findings: tuple[LatentFinding, ...] = ()
    pages: tuple[dict[str, Any], ...] = ()
    root: Path | None = None

    # -- layout -------------------------------------------------------------

    @property
    def fixture_dir(self) -> Path | None:
        """Feed fixture directory of a written dataset (``FeedsConfig.fixture_dir``)."""
        return None if self.root is None else self.root / "feeds"

    @property
    def scans_dir(self) -> Path | None:
        return None if self.root is None else self.root / "scans"

    def scan_paths(self) -> tuple[Path, ...]:
        """Canonical scan documents in scan-date order."""
        directory = self.scans_dir
        if directory is None:
            return ()
        return tuple(directory / f"{scan.scan_id}.json" for scan in self.ordered_scans())

    def ordered_scans(self) -> tuple[Scan, ...]:
        """Scans sorted by date then id: the order a time-ordered split expects."""
        return tuple(sorted(self.scans, key=lambda scan: (scan.scanned_at, scan.scan_id)))

    # -- summary ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Counts a human or a CLI can print without loading anything else."""
        findings = sum(len(scan.findings) for scan in self.scans)
        positives = len(self.oracle.positives())
        dates = sorted(scan.scanned_at.date() for scan in self.scans)
        return {
            "dataset_hash": self.dataset_hash,
            "apps": len({scan.app_id for scan in self.scans}),
            "sectors": sorted({scan.sector for scan in self.scans}),
            "scans": len(self.scans),
            "endpoints": sum(len(scan.endpoints) for scan in self.scans),
            "findings": findings,
            "cves": len(self.world.vulns),
            "kev_fraction": round(self.world.kev_fraction(), 4),
            "reference_pages": len(self.pages),
            "oracle_positives": positives,
            "oracle_positive_rate": round(positives / findings, 4) if findings else 0.0,
            "first_scan": dates[0].isoformat() if dates else None,
            "last_scan": dates[-1].isoformat() if dates else None,
            "root": str(self.root) if self.root is not None else None,
        }

    # -- persistence --------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """``dataset.json``: everything needed to identify and re-locate this world."""
        return {
            "dataset_hash": self.dataset_hash,
            "config": self.config.model_dump(mode="json"),
            "summary": {key: value for key, value in self.summary().items() if key != "root"},
            "scans": [
                {
                    "scan_id": scan.scan_id,
                    "app_id": scan.app_id,
                    "app_name": scan.app_name,
                    "sector": scan.sector,
                    "scanned_at": scan.scanned_at.isoformat(),
                    "findings": len(scan.findings),
                    "path": f"scans/{scan.scan_id}.json",
                }
                for scan in self.ordered_scans()
            ],
        }

    def write(self, output_dir: str | Path | None = None) -> Path:
        """Write the dataset under ``<output_dir>/<dataset hash>/`` and return that root."""
        base = Path(output_dir) if output_dir is not None else DEFAULT_SYNTHETIC_ROOT
        root = base / self.dataset_hash
        (root / "scans").mkdir(parents=True, exist_ok=True)

        for scan in self.ordered_scans():
            _write_json(root / "scans" / f"{scan.scan_id}.json", scan.model_dump(mode="json"))

        write_feed_fixtures(self.world, root / "feeds", pages=self.pages)

        _write_json(root / "world.json", self.world.model_dump(mode="json"))
        _write_json(root / "oracle.json", self.oracle.model_dump(mode="json"))
        _write_json(
            root / "latent_findings.json",
            [item.model_dump(mode="json") for item in self.latent_findings],
        )
        _write_json(root / "dataset.json", self.manifest())
        self.root = root
        return root

    @classmethod
    def load(cls, root: str | Path) -> "SyntheticDataset":
        """Reload a written dataset, validating every document on the way in."""
        base = Path(root)
        manifest = json.loads((base / "dataset.json").read_text(encoding="utf-8"))
        scans = tuple(
            Scan.model_validate(
                json.loads((base / entry["path"]).read_text(encoding="utf-8"))
            )
            for entry in manifest.get("scans", ())
        )
        world = LatentWorld.model_validate(
            json.loads((base / "world.json").read_text(encoding="utf-8"))
        )
        oracle = ExploitationOracle.model_validate(
            json.loads((base / "oracle.json").read_text(encoding="utf-8"))
        )
        latent_path = base / "latent_findings.json"
        latent = (
            tuple(
                LatentFinding.model_validate(item)
                for item in json.loads(latent_path.read_text(encoding="utf-8"))
            )
            if latent_path.exists()
            else ()
        )
        pages_path = base / "feeds" / "references" / "index.json"
        pages = (
            tuple(json.loads(pages_path.read_text(encoding="utf-8")))
            if pages_path.exists()
            else ()
        )
        return cls(
            config=SyntheticConfig.model_validate(manifest.get("config", {})),
            dataset_hash=str(manifest.get("dataset_hash", "")),
            scans=scans,
            world=world,
            oracle=oracle,
            latent_findings=latent,
            pages=pages,
            root=base,
        )

    # -- generation ---------------------------------------------------------

    @classmethod
    def generate(
        cls,
        config: SyntheticConfig | PipelineConfig | None = None,
        output_dir: str | Path | None = None,
        *,
        write: bool = True,
        attacker: AttackerModel | None = None,
    ) -> "SyntheticDataset":
        """Generate (and by default write) a complete synthetic world.

        ``output_dir`` is the *parent* of the dataset directory; the dataset itself lands in
        ``<output_dir>/<dataset hash>/`` so that two configurations never collide and the
        same configuration always resolves to the same path. Pass ``write=False`` to build
        the world in memory without touching the filesystem.
        """
        synthetic = synthetic_config_of(config)
        attacker_model = attacker or _default_attacker()
        seed = int(synthetic.seed)

        start_date = date.fromisoformat(synthetic.start_date)
        interval = max(1, int(synthetic.scan_interval_days))
        scans_per_app = max(1, int(synthetic.scans_per_app))
        stagger = max(7, interval // 2)

        # --- applications and their surfaces
        spec_rng = Random(derive_seed(seed, "apps"))
        specs = generate_app_specs(int(synthetic.n_apps), synthetic.sectors, spec_rng)
        endpoints_by_app: dict[str, tuple[Endpoint, ...]] = {}
        for index, spec in enumerate(specs):
            endpoints_by_app[spec.app_id] = generate_endpoints(
                spec, Random(derive_seed(seed, "endpoints", spec.app_id, index)), synthetic
            )

        # --- the latent vulnerability universe
        last_scan_offset = (len(specs) - 1) * stagger + (scans_per_app - 1) * interval
        horizon_days = int(interval * scans_per_app + 90)
        epss_dates = _snapshot_dates(
            start_date - timedelta(days=365),
            start_date + timedelta(days=last_scan_offset + horizon_days),
            step_days=14,
        )
        stacks = tuple(
            dict.fromkeys(
                (component.vendor or "", component.product, component.version or "0.0.0")
                for spec in specs
                for component in spec.tech_stack
            )
        )
        n_vulns = max(len(VULN_TEMPLATES), int(len(specs) * 14))
        world = build_world(
            n_vulns, stacks, seed, start_date=start_date, epss_dates=epss_dates
        )

        # --- scans and findings
        correlator = FindingCorrelator()
        scans: list[Scan] = []
        latent_findings: list[LatentFinding] = []
        for index, spec in enumerate(specs):
            endpoints = endpoints_by_app[spec.app_id]
            roster = _build_roster(
                spec,
                endpoints,
                world,
                synthetic,
                Random(derive_seed(seed, "roster", spec.app_id)),
            )
            app_start = start_date + timedelta(days=index * stagger)
            for scan_index in range(scans_per_app):
                scanned_at = datetime.combine(
                    app_start + timedelta(days=scan_index * interval),
                    datetime.min.time(),
                ).replace(hour=9, minute=0)
                scan_id = make_scan_id(spec.app_id, scanned_at, spec.scanner_name)
                rng = Random(derive_seed(seed, "scan", spec.app_id, scan_index))
                findings: list[Finding] = []
                for defect in roster:
                    if not (defect.first_scan <= scan_index <= defect.last_scan):
                        continue
                    endpoint = next(
                        (item for item in endpoints if item.endpoint_id == defect.endpoint_id), None
                    )
                    if endpoint is None:
                        continue
                    finding = _make_finding(
                        defect, spec, endpoint, scan_id, scanned_at, world, rng
                    )
                    findings.append(finding)
                    latent_findings.append(
                        LatentFinding(
                            finding_id=finding.finding_id,
                            scan_id=scan_id,
                            app_id=spec.app_id,
                            endpoint_id=endpoint.endpoint_id,
                            defect_key=defect.defect_key,
                            cve_id=defect.cve_id,
                            cwe_id=defect.cwe_id,
                            function=defect.function,
                            observed_at=scanned_at.date(),
                            latent_exposure=defect.latent_exposure,
                            latent_asset_value=defect.latent_asset_value,
                            latent_chain_depth=defect.latent_chain_depth,
                            latent_applies=defect.latent_applies,
                            latent_privilege_gained=defect.privilege_gained,
                        )
                    )
                scan = Scan(
                    scan_id=scan_id,
                    app_id=spec.app_id,
                    app_name=spec.app_name,
                    sector=spec.sector,
                    scanned_at=scanned_at,
                    scanner_name=spec.scanner_name,
                    scanner_version=spec.scanner_version,
                    hosts=spec.hosts,
                    tech_stack=spec.tech_stack,
                    endpoints=endpoints,
                    findings=tuple(findings),
                )
                scans.append(correlator.correlate(scan))

        # --- the latent oracle
        oracle = simulate_exploitation(
            latent_findings,
            world,
            attacker_model,
            seed=seed,
            base_rate=float(synthetic.base_exploit_rate),
            horizon_days=horizon_days,
            lag_bounds=tuple(int(value) for value in synthetic.label_lag_days),
        )

        pages = generate_reference_pages(
            world,
            seed=seed,
            non_english_fraction=float(synthetic.non_english_fraction),
            injection_fraction=float(synthetic.injection_fraction),
        )

        digest = dataset_hash(
            [synthetic.model_dump(mode="json")]
            + [scan.model_dump(mode="json") for scan in scans]
            + [world.model_dump(mode="json"), oracle.model_dump(mode="json")]
        )

        dataset = cls(
            config=synthetic,
            dataset_hash=digest,
            scans=tuple(scans),
            world=world,
            oracle=oracle,
            latent_findings=tuple(latent_findings),
            pages=pages,
        )
        if write:
            dataset.write(output_dir)
        return dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_dates(first: date, last: date, *, step_days: int = 14) -> tuple[date, ...]:
    out: list[date] = []
    cursor = first
    while cursor <= last:
        out.append(cursor)
        cursor += timedelta(days=step_days)
    return tuple(out)


def _make_finding(
    defect: _Defect,
    app: AppSpec,
    endpoint: Endpoint,
    scan_id: str,
    scanned_at: datetime,
    world: LatentWorld,
    rng: Random,
) -> Finding:
    finding_id = make_finding_id(scan_id, endpoint.endpoint_id, defect.plugin_id, defect.param)
    snippet = _EVIDENCE_SNIPPETS.get(defect.cwe_id, "the probe response differed from the control")
    description = (
        f"{defect.title} on {endpoint.method.value} {endpoint.path}. "
        f"The scanner observed that {snippet}."
    )
    evidence = (
        UntrustedText(
            text=f"{endpoint.method.value} {endpoint.url} -> {endpoint.response_status}; {snippet}",
            provenance=Provenance.TARGET_RESPONSE,
            language="en",
        ),
    )
    affected: TechComponent | None = None
    if defect.affected_product:
        affected = next(
            (item for item in app.tech_stack if item.product == defect.affected_product), None
        )
    elif defect.cve_id:
        vuln = world.by_id(defect.cve_id)
        if vuln is not None:
            affected = next(
                (item for item in app.tech_stack if item.product == vuln.product), None
            )
    return Finding(
        finding_id=finding_id,
        scan_id=scan_id,
        app_id=app.app_id,
        endpoint_id=endpoint.endpoint_id,
        name=defect.name,
        cwe_id=defect.cwe_id,
        cve_ids=(defect.cve_id,) if defect.cve_id else (),
        scanner=app.scanner_name,
        scanner_plugin_id=defect.plugin_id,
        scanner_severity=defect.severity,
        scanner_confidence=defect.confidence,
        description=UntrustedText(
            text=description, provenance=Provenance.SCANNER_OUTPUT, language="en"
        ),
        evidence=evidence,
        affected_component=affected,
        observed_at=scanned_at,
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return path
