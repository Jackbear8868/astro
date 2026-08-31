"""The six standalone steps, chained from a config file.

    python run_pipeline.py ../../../configs/p01.yaml

It reads the same config `pipeline.py` does and calls the step functions in this folder
in the same order, so a run started here and a run started there are the same run. What
it does not have is the live pipeline's per-step logs and terminal filtering: this is
the six scripts one after another, and it exists so the chain can be exercised without
going through the live entrance.

The steps hand their results on in memory here, exactly as the live pipeline does. Each
`stepN_*.py` can also be run alone from the command line, and then it reads what it
needs from an earlier step's products instead -- see `step_io.py`.
"""

import argparse
import time
from pathlib import Path

import config as cfg_mod
from step1_whitelight import place_segmentation, whitelight
from step2_object_spectra import source_spectra
from step3_sky_basis import sky_basis
from step4_classify_sources import classify_sources
from step5_fit_s_field import fit_sky_amplitude
from step6_subtract_sky import subtract_sky


# A bound past any real image, standing in for "no bound on that side". The steps take
# a number, and this is what a null in the config becomes.
BEYOND_EDGE = 9999


def region(cfg, which):
    """The sky_region box as the step that uses it takes it.

    One box in the config serves two steps, and `apply_to` says which. A step not named
    there gets nothing, which is how "restrict the basis but not the s field" is
    written with one box.

    Config ranges are half-open with null for "no bound", and xlim/ylim have the same
    meaning; exclude_box includes both endpoints, so its upper bound loses one.
    """
    reg = cfg["sky_region"]
    if which not in reg["apply_to"]:
        return {}
    x, y = reg["x"], reg["y"]
    lo = lambda v: 0 if v is None else v

    if reg["include"]:
        kw = {}
        if x != [None, None]:
            kw["xlim"] = [lo(x[0]), BEYOND_EDGE if x[1] is None else x[1]]
        if y != [None, None]:
            kw["ylim"] = [lo(y[0]), BEYOND_EDGE if y[1] is None else y[1]]
        return kw

    return {"exclude_box": [
        lo(y[0]), BEYOND_EDGE if y[1] is None else y[1] - 1,
        lo(x[0]), BEYOND_EDGE if x[1] is None else x[1] - 1]}


def run_pointing(cfg_path):
    """Run one pointing's config through the six steps in this folder."""
    cfg = cfg_mod.load(cfg_path)
    inp, out = cfg["input"], cfg["output"]
    for key, path in inp.items():
        if not Path(path).exists():
            raise SystemExit(f"★ input.{key} not found: {path}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    b, s, a, sp = (cfg["sky_line_basis"], cfg["source_fit"],
                   cfg["sky_amplitude"], cfg["spaxel_fit"])
    basis_reg = region(cfg, "basis")
    train_reg = region(cfg, "sky_amplitude")

    print("=" * 70)
    print(f"  pointing #{cfg['pointing']}  ->  {out}   [{Path(cfg_path).name}]")
    print("=" * 70)
    t0 = time.time()

    print("--- [1/6] step1 white light (from the nosky cube), and the segmentation")
    white = whitelight(inp["nosky"], out / "step01")
    seg = place_segmentation(white, inp["seg"], out / "step01", inp["nosky"],
                             max_offset=cfg["max_grid_offset"])

    print("--- [2/6] step2 source spectra (nosky, for classification)")
    spectra = source_spectra(white, seg, inp["nosky"], out / "step02")

    print("--- [3/6] step3 sky basis")
    sky = sky_basis(white, seg, inp["cube"], out, out / "step03",
                    K=b["K"], methods=[b["method"]], seed=b["seed"],
                    continuum_window=b["continuum_window"],
                    line_thresholds=tuple(b["line_thresholds"]),
                    max_iter=b["max_iter"], clip_sigma=b["clip_sigma"],
                    min_unmasked_frac=b["min_unmasked_frac"],
                    xlim=basis_reg.get("xlim"), ylim=basis_reg.get("ylim"),
                    exclude_box=basis_reg.get("exclude_box"),
                    borrow_from=b.get("borrow_from"),
                    mask_source_lines=b.get("mask_source_lines"),
                    select_faintest=b.get("select_faintest"))

    print("--- [4/6] step4 template fitting and classification")
    classified = classify_sources(
        sky, spectra, out, inp["nosky"], seg.path,
        K=b["K"], basis=b["method"], fit_window=tuple(s["fit_window"]),
        line_mask_iter=tuple(s["line_mask_iter"]), fix_s_at=s["fix_s_at"],
        z_min=s["z_min"], z_max=s["z_max"], z_step=s["z_step"],
        star_dz=s["star_dz"], num_workers=s["num_workers"],
        keep_scans=s["keep_scans"])

    print(f"--- [5/6] step5 build the s field   "
          f"[mask iter {s['line_mask_iter'][-1]}]")
    s_field = fit_sky_amplitude(
        white, seg, sky, classified, inp["cube"], out, out / "step05",
        basis=b["method"], K=b["K"],
        blank_channels=sp["blank_channels"],
        min_channel_coverage=sp["min_channel_coverage"],
        min_source_distance=a["min_source_distance"],
        min_main_source_distance=a["min_main_source_distance"],
        train_clip_sigma=a["train_clip_sigma"],
        main_source_dz=a["main_source_dz"], n_iter=a["n_iter"],
        train_xlim=train_reg.get("xlim"), train_ylim=train_reg.get("ylim"),
        train_exclude_box=train_reg.get("exclude_box"))

    print("--- [6/6] step6 final sky subtraction")
    subtract_sky(white, seg, sky, classified, s_field, inp["cube"], out,
                 out / "step06", basis=b["method"], K=b["K"],
                 blank_channels=sp["blank_channels"],
                 min_channel_coverage=sp["min_channel_coverage"])

    print(f"*** pointing #{cfg['pointing']} done in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(
        description="run the standalone steps in order, from a pointing config")
    ap.add_argument("config", nargs="+", help="pointing config file(s)")
    args = ap.parse_args()
    for p in args.config:
        run_pointing(p)


if __name__ == "__main__":
    main()
