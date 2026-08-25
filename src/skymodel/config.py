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
APPLY_TO = ("basis", "sky_amplitude")
# The channels the blank spaxels are solved on: every channel, or only those left
# by step3's first sky-line mask. step5 and step6 import this for their own
# --blank-channels, so the two ways in accept the same words.
BLANK_CHANNELS = ("all", "line1")
SECTIONS = ("input", "sky_region", "sky_line_basis", "source_fit",
            "sky_amplitude", "spaxel_fit")

# How far apart the seg and white-light grids may be before the pointing is
# refused, in pixels. The 13 pointings that pass sit at 0.000-0.020 px, so this is
# loose by more than an order of magnitude and still catches a real mismatch.
# Optional in the file: a pointing writes it only to raise it, and raising it is
# a decision to run on headers that disagree.
MAX_GRID_OFFSET = 0.1


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


def _number(v, name, ge=None, gt=None, le=None):
    """A number, optionally bounded: >= ge, > gt, <= le.

    Only bounds that hold whatever the data looks like belong here -- a distance
    cannot be negative, a fraction of the channels cannot exceed 1. How far apart
    two things should be, or how hard to clip, is not a question this file can
    answer. A missing key arrives as None and fails as "not a number", which is
    what it is.
    """
    if not isinstance(v, (int, float)):
        _fail(f"{name} must be a number, got {v!r}")
    if ge is not None and v < ge:
        _fail(f"{name} must be >= {ge}, got {v!r}")
    if gt is not None and v <= gt:
        _fail(f"{name} must be > {gt}, got {v!r}")
    if le is not None and v > le:
        _fail(f"{name} must be <= {le}, got {v!r}")
    return v


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
    # Iterations are numbered from 1, and step4 reads them as line_masks[it - 1]:
    # 0 or less indexes backwards from the end and quietly fits a different mask.
    # There is no upper bound here -- how many iterations exist is only known once
    # step3's masks are on disk, so step4 is where that half of it is checked.
    for it in s["line_mask_iter"]:
        if not isinstance(it, int) or it < 1:
            _fail("source_fit.line_mask_iter: iterations are integers counting "
                  f"from 1, got {it!r}")

    a = cfg["sky_amplitude"]
    _number(a.get("min_source_distance"),
            "sky_amplitude.min_source_distance", ge=0)
    _number(a.get("min_main_source_distance"),
            "sky_amplitude.min_main_source_distance", ge=0)
    _number(a.get("train_clip_sigma"), "sky_amplitude.train_clip_sigma", gt=0)
    _number(a.get("main_source_dz"), "sky_amplitude.main_source_dz", ge=0)

    sp = cfg["spaxel_fit"]
    if sp.get("blank_channels") not in BLANK_CHANNELS:
        _fail(f"spaxel_fit.blank_channels must be one of {list(BLANK_CHANNELS)}, "
              f"got {sp.get('blank_channels')!r}")
    # A fraction of the wavelength channels, so it lives in [0, 1]. Written as a
    # channel count instead, no spaxel could ever reach it and none would be fitted.
    _number(sp.get("min_channel_coverage"),
            "spaxel_fit.min_channel_coverage", ge=0, le=1)

    # Optional, so it is filled in here rather than required of all 14 files. It
    # still lives only in the config: a limit raised on the command line would be
    # a bypass that leaves no record of which pointing it was applied to.
    g = cfg.setdefault("max_grid_offset", MAX_GRID_OFFSET)
    if not isinstance(g, (int, float)) or g <= 0:
        _fail(f"max_grid_offset must be a positive number, got {g!r}")

    return cfg
