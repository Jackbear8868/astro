"""Read one pointing's configuration. Everything the pipeline needs lives in one YAML
file (configs/pNN.yaml); load() reads it, checks the values and returns a nested dict
with the paths resolved to absolute Paths. Pixel ranges are written [lo, hi] and read
half-open, lo <= i < hi, and either end may be null.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

METHODS  = ("svd", "pca")
APPLY_TO = ("basis", "sky_amplitude")
# Channels the blank spaxels are solved on: all, or step3's first sky-line mask.
BLANK_CHANNELS = ("all", "line1")
SECTIONS = ("input", "sky_region", "sky_line_basis", "source_fit",
            "sky_amplitude", "spaxel_fit")

# How far apart the seg and white-light grids may be before the pointing is refused,
# in pixels. Optional: raising it is a decision to run on headers that disagree.
MAX_GRID_OFFSET = 0.1

# Whether steps 1-5 leave their products on disk; step6's are written either way.
KEEP_INTERMEDIATE = True

# Whether step4 keeps the whole chi2 surface it scanned, one file per branch. Turning it
# off leaves no record of the search, though source_fits.npz still holds the best fits.
KEEP_SCANS = True


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
    """A number, optionally bounded: >= ge, > gt, <= le. Only bounds that hold whatever
    the data looks like belong here -- a distance cannot be negative, a fraction cannot
    exceed 1. How far apart two things should be, this file cannot answer.
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
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        _fail(f"{path} does not contain a mapping")

    for k in ("pointing", "output", *SECTIONS):
        if k not in cfg:
            _fail(f"{path.name} has no '{k}' key")
    if not isinstance(cfg["pointing"], int):
        _fail(f"pointing must be an integer, got {cfg['pointing']!r}")

    # Stored relative to the repository root so a config is readable from anywhere.
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
    # A fraction of the channels: at 1 no mask is acceptable, at 0 every mask is.
    _number(b.get("min_unmasked_frac"),
            "sky_line_basis.min_unmasked_frac", ge=0, le=1)
    # Optional: writing it takes the line basis from another finished run instead of
    # learning one here, and names that run's output directory. Absent and null both
    # mean the basis is learned from this pointing, so an absent key is left absent.
    src = b.get("borrow_from")
    if src is not None:
        if not isinstance(src, str):
            _fail("sky_line_basis.borrow_from must be the output directory of another "
                  f"run, or null, got {src!r}")
        b["borrow_from"] = ROOT / src
    # Optional: writing it keeps the source's own emission lines out of the basis. The
    # windows are excluded from step3's decomposition and from nothing else -- the
    # continuum, the mean spectrum and the sky-line masks still see every channel. It is
    # written either as a list of [low, high] observed windows or as a mapping of rest
    # wavelengths and a half width, normalised here to keep the redshift out of step3.
    ml = b.get("mask_source_lines")
    if isinstance(ml, list):
        if not ml:
            _fail("sky_line_basis.mask_source_lines: the window list is empty; write "
                  "null, or leave the key out, to exclude nothing")
        windows = []
        for w in ml:
            if not (isinstance(w, (list, tuple)) and len(w) == 2):
                _fail("sky_line_basis.mask_source_lines: each window is a [low, high] "
                      f"pair of observed wavelengths in Angstrom, got {w!r}")
            lo, hi = w
            _number(lo, "sky_line_basis.mask_source_lines window low", gt=0)
            _number(hi, "sky_line_basis.mask_source_lines window high", gt=0)
            # Ordered, so of non-zero width: [high, low] would select no channel at all.
            if float(hi) <= float(lo):
                _fail("sky_line_basis.mask_source_lines: window "
                      f"[{lo}, {hi}] is not low < high")
            windows.append([float(lo), float(hi)])
        b["mask_source_lines"] = windows
    elif ml is not None:
        if not isinstance(ml, dict):
            _fail("sky_line_basis.mask_source_lines must be a list of [low, high] "
                  "observed windows, or a mapping with redshift, rest_wavelengths and "
                  f"half_width, or null, got {ml!r}")
        unknown = sorted(set(ml) - {"redshift", "rest_wavelengths", "half_width"})
        if unknown:
            _fail(f"sky_line_basis.mask_source_lines: unknown key(s) {unknown}")
        for k in ("redshift", "rest_wavelengths", "half_width"):
            if k not in ml:
                _fail(f"sky_line_basis.mask_source_lines.{k} is missing")
        # A redshift, so > -1; how far away the source is, is not decided here.
        _number(ml["redshift"], "sky_line_basis.mask_source_lines.redshift", gt=-1)
        # Half a window in Angstrom, so positive: zero width would exclude nothing.
        _number(ml["half_width"], "sky_line_basis.mask_source_lines.half_width", gt=0)
        rest = ml.get("rest_wavelengths")
        if not isinstance(rest, list) or not rest:
            _fail("sky_line_basis.mask_source_lines.rest_wavelengths must be a "
                  f"non-empty list of rest wavelengths in Angstrom, got {rest!r}")
        for w in rest:
            _number(w, "sky_line_basis.mask_source_lines.rest_wavelengths", gt=0)
        z = float(ml["redshift"])
        hw = float(ml["half_width"])
        b["mask_source_lines"] = [[float(w) * (1 + z) - hw, float(w) * (1 + z) + hw]
                                  for w in rest]
    # Optional: writing it picks the spaxels the sky is learned from by flux rather than
    # by the segmentation alone -- the ESO pipeline's own rule, which ranks the field,
    # discards the faintest `ignore` and learns from the next `fraction`. Absent = all.
    sf = b.get("select_faintest")
    if sf is not None:
        if not isinstance(sf, dict):
            _fail("sky_line_basis.select_faintest must be a mapping with ignore and "
                  f"fraction, or null, got {sf!r}")
        unknown = sorted(set(sf) - {"ignore", "fraction"})
        if unknown:
            _fail(f"sky_line_basis.select_faintest: unknown key(s) {unknown}")
        for k in ("ignore", "fraction"):
            if k not in sf:
                _fail(f"sky_line_basis.select_faintest.{k} is missing")
        # Both are fractions of the ranked field, so in [0, 1]. fraction is above 0
        # because a zero-width window leaves no spaxel to learn from; ignore may be 0.
        _number(sf["ignore"], "sky_line_basis.select_faintest.ignore", ge=0, le=1)
        _number(sf["fraction"], "sky_line_basis.select_faintest.fraction", gt=0, le=1)
        # The window ends at the `ignore + fraction` percentile, and none runs past 100.
        if sf["ignore"] + sf["fraction"] > 1:
            _fail("sky_line_basis.select_faintest: ignore + fraction is the top of the "
                  f"window and cannot exceed 1, got {sf['ignore']} + {sf['fraction']}")

    s = cfg["source_fit"]
    s["fit_window"] = _pair(s.get("fit_window"), "source_fit.fit_window")
    if s["fit_window"][0] >= s["fit_window"][1]:
        _fail(f"source_fit.fit_window must increase, got {s['fit_window']}")
    if not isinstance(s.get("line_mask_iter"), list) or not s["line_mask_iter"]:
        _fail("source_fit.line_mask_iter must be a non-empty list of iteration numbers")
    # Iterations count from 1: step4 reads line_masks[it - 1], so 0 or less would index
    # backwards from the end. The upper bound is step4's, once step3 has run.
    for it in s["line_mask_iter"]:
        if not isinstance(it, int) or it < 1:
            _fail("source_fit.line_mask_iter: iterations are integers counting "
                  f"from 1, got {it!r}")
    # Optional: a config writes it only to turn the scans off.
    s.setdefault("keep_scans", KEEP_SCANS)
    if not isinstance(s["keep_scans"], bool):
        _fail(f"source_fit.keep_scans must be true or false, got {s['keep_scans']!r}")

    a = cfg["sky_amplitude"]
    _number(a.get("min_source_distance"),
            "sky_amplitude.min_source_distance", ge=0)
    _number(a.get("min_main_source_distance"),
            "sky_amplitude.min_main_source_distance", ge=0)
    _number(a.get("train_clip_sigma"), "sky_amplitude.train_clip_sigma", gt=0)
    _number(a.get("main_source_dz"), "sky_amplitude.main_source_dz", ge=0)
    if not isinstance(a.get("n_iter"), int) or a["n_iter"] < 1:
        _fail(f"sky_amplitude.n_iter must be a positive integer, "
              f"got {a.get('n_iter')!r}")

    sp = cfg["spaxel_fit"]
    if sp.get("blank_channels") not in BLANK_CHANNELS:
        _fail(f"spaxel_fit.blank_channels must be one of {list(BLANK_CHANNELS)}, "
              f"got {sp.get('blank_channels')!r}")
    # A fraction of the channels, in [0, 1]; a channel count would be reached by none.
    _number(sp.get("min_channel_coverage"),
            "spaxel_fit.min_channel_coverage", ge=0, le=1)

    # Filled in rather than required of every config; raising it records the pointing.
    g = cfg.setdefault("max_grid_offset", MAX_GRID_OFFSET)
    if not isinstance(g, (int, float)) or g <= 0:
        _fail(f"max_grid_offset must be a positive number, got {g!r}")

    # Filled in for the same reason; turning it off is a decision to keep nothing
    # but step6's cubes.
    k = cfg.setdefault("keep_intermediate", KEEP_INTERMEDIATE)
    if not isinstance(k, bool):
        _fail(f"keep_intermediate must be true or false, got {k!r}")

    return cfg
