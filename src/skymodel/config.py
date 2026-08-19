"""Read one pointing's configuration.

Everything the pipeline needs for a pointing lives in a single YAML file
(configs/pNN.yaml). load() reads that file, checks the values and returns a
plain nested dict, with the paths resolved to absolute Paths so that callers
never join paths themselves.

Pixel ranges are written [lo, hi] and read half-open: lo <= i < hi. Either end
may be null, which means "no bound on that side" -- writing a number there that
is meant to be "the edge of the field" would have to be right for all 14
pointings, and a number that is too small silently drops part of the region.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

METHODS  = ("svd", "pca")
APPLY_TO = ("basis", "s_field")
SECTIONS = ("input", "sky_region", "sky_line_basis", "source_fit", "s_field",
            "spaxel_fit")


def _fail(msg):
    raise SystemExit(f"★ config: {msg}")


def _axis(v, name):
    """One axis of sky_region: [lo, hi], half-open, either end may be null."""
    if not isinstance(v, list) or len(v) != 2:
        _fail(f"sky_region.{name} must be a two-element list [lo, hi]")
    lo, hi = v
    for e in (lo, hi):
        if e is not None and not isinstance(e, int):
            _fail(f"sky_region.{name}: bounds are integers or null, got {e!r}")
    if lo is not None and hi is not None and lo >= hi:
        _fail(f"sky_region.{name}: the range is half-open, so lo < hi; got {v}")
    return [lo, hi]


def _pair(v, name):
    """A two-element list of numbers."""
    ok = (isinstance(v, list) and len(v) == 2
          and all(isinstance(e, (int, float)) for e in v))
    if not ok:
        _fail(f"{name} must be a two-element list of numbers, got {v!r}")
    return list(v)


def load(path):
    """Read a pointing config and return the checked dict."""
    path = Path(path)
    if not path.exists():
        _fail(f"file not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        _fail(f"{path} does not contain a mapping")

    for k in ("pointing", "output", *SECTIONS):
        if k not in cfg:
            _fail(f"{path.name} has no '{k}' key")
    if not isinstance(cfg["pointing"], int):
        _fail(f"pointing must be an integer, got {cfg['pointing']!r}")

    # Paths are stored relative to the repository root so that a config is
    # readable from anywhere; they are handed out absolute.
    inp = cfg["input"]
    for k in ("cube", "nosky", "seg"):
        if k not in inp:
            _fail(f"input.{k} is missing")
        inp[k] = ROOT / inp[k]
    cfg["output"] = ROOT / cfg["output"]

    reg = cfg["sky_region"]
    reg["x"] = _axis(reg.get("x"), "x")
    reg["y"] = _axis(reg.get("y"), "y")
    if not isinstance(reg.get("include"), bool):
        _fail("sky_region.include must be true or false")
    if not isinstance(reg.get("apply_to"), list):
        _fail(f"sky_region.apply_to must be a list; allowed entries {list(APPLY_TO)}")
    unknown = sorted(set(reg["apply_to"]) - set(APPLY_TO))
    if unknown:
        _fail(f"sky_region.apply_to: unknown {unknown}; allowed {list(APPLY_TO)}")

    b = cfg["sky_line_basis"]
    if b.get("method") not in METHODS:
        _fail(f"sky_line_basis.method must be one of {list(METHODS)}, "
              f"got {b.get('method')!r}")
    if not isinstance(b.get("K"), int) or b["K"] < 1:
        _fail(f"sky_line_basis.K must be a positive integer, got {b.get('K')!r}")
    b["line_thresholds"] = _pair(b.get("line_thresholds"),
                                 "sky_line_basis.line_thresholds")

    s = cfg["source_fit"]
    s["fit_window"] = _pair(s.get("fit_window"), "source_fit.fit_window")
    if s["fit_window"][0] >= s["fit_window"][1]:
        _fail(f"source_fit.fit_window must increase, got {s['fit_window']}")
    if not isinstance(s.get("line_mask_iter"), list) or not s["line_mask_iter"]:
        _fail("source_fit.line_mask_iter must be a non-empty list of iteration numbers")

    return cfg
