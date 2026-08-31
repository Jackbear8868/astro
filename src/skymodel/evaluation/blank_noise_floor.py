"""Is what is left in blank noise, or is it wrong? -- per channel, for both pipelines.

A mean over tens of thousands of blank spaxels looks flat whether the subtraction was
good or bad, as long as the error is random: averaging N spaxels divides random scatter
by sqrt(N) and leaves a systematic offset untouched. So each panel draws the mean
residual against its own noise floor:

    scatter   the spread across blank spaxels within one channel, (p84 - p16) / 2 --
              what a single spaxel actually looks like.
    floor     scatter / sqrt(N). Inside this band the channel's residual is
              indistinguishable from the same spaxels averaged with no systematic
              error; outside it, the same mistake was made in every spaxel.

The band is what makes the two pipelines comparable: rms and mean both shrink with the
number of spaxels averaged, so they say as much about the size of the blank region as
about the sky model, while the ratio to the floor does not. The top panel draws the two
scatters together, so how much of the separation below is systematic can be judged.

    conda run -n astro python src/skymodel/evaluation/blank_noise_floor.py --work results/skymodel/p01 \\
        --n-floor 3 --ylim -0.4 0.4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402
from config import resolve_path  # noqa: E402
from products import Run  # noqa: E402
from common import data_hdu  # noqa: E402
from products import latest_run  # noqa: E402
from spectra import robust_range  # noqa: E402
from zones import blank_mask  # noqa: E402

C_OURS, C_ESO, C_BAND, C_ZERO = "#1f77b4", "#e8710a", "0.72", "0.45"
CHUNK = 200


def channel_stats(hdu, mask, keep, nz):
    """Per channel: the mean across the kept blank spaxels, and their robust scatter.

    (p84 - p16) / 2 rather than the standard deviation: a handful of bad spaxels would
    set the standard deviation, and a floor built from it would be so wide that nothing
    could ever leave the band.
    """
    mean = np.empty(nz)
    scat = np.empty(nz)
    for j in range(0, nz, CHUNK):
        x = np.asarray(hdu.data[j:j + CHUNK], np.float32)[:, mask][:, keep]
        mean[j:j + x.shape[0]] = x.mean(axis=1, dtype=np.float64)
        p16, p84 = np.percentile(x, [16, 84], axis=1)
        scat[j:j + x.shape[0]] = (p84 - p16) / 2
        print(f"    {min(j + CHUNK, nz)}/{nz}", end="\r", flush=True)
    print(" " * 24, end="\r")
    return mean, scat


def main():
    ap = argparse.ArgumentParser(
        description="Blank residual against the noise floor, per channel, ours vs ESO")
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=None,
                    help="glob naming the run under step05 that holds our sky_subtracted.fits")
    ap.add_argument("--nosky", default=None)
    ap.add_argument("--n-floor", type=float, default=1.0,
                    help="width of the drawn band in noise floors. 1 is 'the mean of "
                         "this many spaxels with no systematic error'")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y range of the two residual panels, shared")
    ap.add_argument("--scatter-ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(22, 11))
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pointing = Run(args.work)
    W = pointing.work
    name = pointing.name
    meta = pointing.meta(3)
    run = latest_run(W, "sky_subtracted.fits", "step06", args.run)
    if run is None:
        raise SystemExit(f"no sky_subtracted.fits under {W}/step05 or {W}/step06")
    # ESO's cube as the run's config names it. Deriving it from the wsky filename,
    # which is what this did, only ever finds data kept inside the repository.
    nosky = resolve_path(args.nosky) if args.nosky else pointing.nosky
    if not nosky.exists():
        raise SystemExit(f"{nosky} does not exist")

    m, n_all, _ = blank_mask(W, meta)
    wl = pointing.wl
    nz = wl.size
    print(f"{name}:  ours {run.relative_to(ROOT)}   ESO {nosky.name}")

    with fits.open(run / "sky_subtracted.fits", memmap=True) as ha, \
         fits.open(nosky, memmap=True) as he:
        da, de = data_hdu(ha), data_hdu(he)
        ca = np.ones(int(m.sum()), bool)
        ce = np.ones(int(m.sum()), bool)
        for j in range(0, nz, CHUNK):
            ca &= np.isfinite(np.asarray(da.data[j:j + CHUNK], np.float32)[:, m]).all(axis=0)
            ce &= np.isfinite(np.asarray(de.data[j:j + CHUNK], np.float32)[:, m]).all(axis=0)
            print(f"    coverage {min(j + CHUNK, nz)}/{nz}", end="\r", flush=True)
        print(" " * 30, end="\r")
        keep = ca & ce
        N = int(keep.sum())
        print(f"  blank {n_all:,} -> {int(m.sum()):,} in the step3 mask -> "
              f"{N:,} complete in both cubes   sqrt(N) = {np.sqrt(N):.1f}")
        mo, so = channel_stats(da, m, keep, nz)
        me, se = channel_stats(de, m, keep, nz)

    fo, fe = so / np.sqrt(N), se / np.sqrt(N)
    print(f"\n    {'':<6}{'scatter':>10}{'floor':>10}{'|mean|/floor':>15}"
          f"{'channels > ' + str(args.n_floor) + 'x':>18}")
    for lab, mm, sc, fl in (("ours", mo, so, fo), ("ESO", me, se, fe)):
        r = np.abs(mm) / fl
        print(f"    {lab:<6}{np.median(sc):>10.3f}{np.median(fl):>10.4f}"
              f"{np.median(r):>15.2f}"
              f"{f'{int((r > args.n_floor).sum()):,} / {nz:,}':>18}"
              f"  ({100 * (r > args.n_floor).mean():.1f}%)")

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=args.figsize,
                           gridspec_kw={"height_ratios": [1, 1.3, 1.3], "hspace": 0.09})

    ax[0].plot(wl, so, lw=0.6, color=C_OURS, label="ours")
    ax[0].plot(wl, se, lw=0.6, color=C_ESO, label="ESO")
    ax[0].set_ylabel("scatter across\nblank spaxels")
    ax[0].legend(fontsize=11, loc="upper left", frameon=False)
    ax[0].grid(alpha=0.2)
    if args.scatter_ylim:
        ax[0].set_ylim(*args.scatter_ylim)
    else:
        ax[0].set_ylim(0, np.percentile(np.concatenate([so, se]), 99.5) * 1.15)

    # The two residual panels share a y range, or the larger residual is squeezed to
    # look like the smaller one.
    lim = args.ylim if args.ylim else robust_range(np.concatenate([mo, me]))
    for a, (lab, mm, fl, c) in zip(ax[1:], (("ours", mo, fo, C_OURS),
                                            ("ESO", me, fe, C_ESO))):
        a.fill_between(wl, -args.n_floor * fl, args.n_floor * fl, color=C_BAND,
                       lw=0, label=f"$\\pm${args.n_floor:g} $\\times$ noise floor")
        a.axhline(0, lw=0.8, color=C_ZERO)
        a.plot(wl, mm, lw=0.6, color=c, label=f"{lab}: mean over blank")
        a.set_ylabel("flux")
        a.set_ylim(*lim)
        a.legend(fontsize=11, loc="upper left", frameon=False, ncol=2)
        a.grid(alpha=0.2)
    ax[2].set_xlabel("wavelength [$\\AA$]")
    ax[2].set_xlim(wl.min(), wl.max())

    out = (Path(args.out) if args.out
           else pointing.figdir("sky") / f"blank_noise_floor_{run.name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
