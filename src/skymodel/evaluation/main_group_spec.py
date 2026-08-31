"""For every member of the adjacent blob, use the spectrum to judge whether it is part
of the main galaxy.

SExtractor's deblender splits the main galaxy into several pieces, and "they are stuck
together" is not enough to group them -- another object superimposed on the galaxy is
stuck to it just as well. This script lists the spectral evidence for that judgement,
one table per pointing:

  star / gal rchi2   the best reduced chi2 of each branch of the same source. Only
                     comparable within one row: a bright source has little photon
                     noise, so a slight model imperfection over a small sigma is
                     already a large chi2.
  z_gal, dv          the best redshift of the galaxy branch, and its velocity
                     difference from the main source group, whose redshift is the one
                     of the member containing the brightest pixel.
  n_z                grid points in the galaxy branch within +1% of the best rchi2 --
                     how tightly the redshift is pinned down.
  forced             by how many % the rchi2 worsens when this member is forced onto
                     the main source group's redshift.

    conda run -n astro python src/skymodel/evaluation/main_group_spec.py -n 1 4 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402
from products import Run  # noqa: E402
from utils import DZ_MAX, load_scan  # noqa: E402

C_KMS = 299792.458


def blob_members(seg, white):
    """The connected component containing the brightest pixel, and all the seg IDs
    inside it (with no area filtering at all)."""
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(white), white, -np.inf)),
                         white.shape)
    lab, _ = ndimage.label(seg > 0)
    blob = lab == lab[k]
    ids = [int(i) for i in np.unique(seg[blob]) if i > 0]
    return ids, blob, k


def area_keep(seg, ids, min_frac):
    """The area criterion, kept for comparison with the redshift criterion utils
    uses."""
    area = {i: int((seg == i).sum()) for i in ids}
    top = max(area.values())
    return [i for i in ids if area[i] >= min_frac * top]


def branch_best(work, sid):
    """The best solution of each of the two branches of one seg ID.

    step4 stores them as step04/scans_star.npz and scans_galaxy.npz; on the star side
    z is a radial velocity. Returns (best star rchi2, best galaxy rchi2, best galaxy z,
    the galaxy's z and rchi2 arrays).
    """
    step04 = work / "step04"
    try:
        d_star = load_scan(step04, "star", sid)
        d_gal  = load_scan(step04, "galaxy", sid)
    except SystemExit:
        return None
    r_star = float(d_star["red_chi2"].min())
    z, r = d_gal["z"], d_gal["red_chi2"]
    j = int(np.argmin(r))
    return r_star, float(r[j]), float(z[j]), z, r


def main():
    ap = argparse.ArgumentParser(description="Spectral evidence for adjacent members")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)))
    ap.add_argument("--min-frac", type=float, default=0.05,
                    help="area-criterion threshold for comparison")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="tolerance fraction relative to the best rchi2 when counting n_z")
    ap.add_argument("--dz-max", type=float, default=DZ_MAX,
                    help="spectral criterion: max redshift difference from the main source to count as part of the galaxy")
    args = ap.parse_args()

    rows = []
    for n in args.n:
        pointing = Run(ROOT / f"results/skymodel/p{n:02d}")
        W = pointing.work
        seg, white, valid = pointing.seg, pointing.white, pointing.valid
        wn = np.where(valid, white, np.nan)
        ids, _, k = blob_members(seg, wn)
        keep_area = area_keep(seg, ids, args.min_frac)

        # the brightest pixel, not the largest area, marks the core
        peak_id = int(seg[k])
        # The table needs the whole chi2 surface of both branches, which step4 keeps
        # only with source_fit.keep_scans on; without it every column below is empty.
        if not (W / "step04/scans_galaxy.npz").exists():
            raise SystemExit(
                f"★ {W / 'step04/scans_galaxy.npz'} does not exist. This table needs the whole "
                "chi2 scan of each source, which step4 writes only when "
                "source_fit.keep_scans is true in the config. Set it and re-run step4 "
                "for this pointing.")
        b = branch_best(W, peak_id)
        if b is None:
            print(f"p{n:02d}: step04 has no scan file for id{peak_id}, skipping")
            continue
        z_main = b[2]

        print(f"\n=== p{n:02d}   main source z = {z_main:.4f}"
              f" (from the brightest pixel, id {peak_id}) ===")
        print(f"{'id':>5} {'area':>8} {'star rchi2':>12} {'gal rchi2':>12} {'gal/star':>9}"
              f" {'z_gal':>8} {'dv km/s':>10} {'n_z':>6} {'forced':>8}"
              f"  area crit  spec crit")
        print("-" * 112)
        for i in ids:
            b = branch_best(W, i)
            if b is None:
                print(f"{i:>5}  -- step04 has no scan file for this ID --")
                continue
            r_star, r_gal, z_gal, z, r = b
            area = int((seg == i).sum())
            dz = z_gal - z_main
            dv = C_KMS * dz / (1 + z_main)
            n_z = int((r <= r_gal * (1 + args.tol)).sum())
            # forced onto the main group's redshift: the grid point closest to z_main
            jf = int(np.argmin(np.abs(z - z_main)))
            forced = 100 * (r[jf] - min(r_gal, r_star)) / min(r_gal, r_star)
            verdict = "keep" if i in keep_area else "drop"
            v_spec = "keep" if abs(dz) <= args.dz_max else "drop"
            flag = "" if verdict == v_spec else "   <<< criteria disagree"
            print(f"{i:>5} {area:>8,} {r_star:>12,.2f} {r_gal:>12,.2f}"
                  f" {r_gal/r_star:>9.3f} {z_gal:>8.4f} {dv:>+10,.0f} {n_z:>6,}"
                  f" {forced:>+7.1f}%  {verdict:>6}  {v_spec:>6}{flag}")
            rows.append((n, i, area, r_star, r_gal, z_gal, dv, n_z, forced,
                         i in keep_area, dz))

    print(f"\n\n=== all {len(rows)} adjacent members ===")
    dv = np.array([r[6] for r in rows])
    nz = np.array([r[7] for r in rows])
    ratio = np.array([r[4]/r[3] for r in rows])
    kept = np.array([r[9] for r in rows], bool)
    print(f"area criterion keeps {int(kept.sum())}, drops {int((~kept).sum())}")
    print(f"|dv| distribution:  <100 km/s {int((np.abs(dv) < 100).sum())},"
          f"  100-1000 {int(((np.abs(dv) >= 100) & (np.abs(dv) < 1000)).sum())},"
          f"  >1000 {int((np.abs(dv) >= 1000).sum())}")
    print(f"n_z distribution:   =1 {int((nz == 1).sum())},"
          f"  2-10 {int(((nz > 1) & (nz <= 10)).sum())},"
          f"  >10 {int((nz > 10).sum())}")
    print(f"gal/star:   <1 (galaxy fits better) {int((ratio < 1).sum())},"
          f"  >=1 (star fits better) {int((ratio >= 1).sum())}")

    print(f"\nmembers where the two criteria disagree (dz_max = {args.dz_max:g}):")
    bad = [r for r in rows if r[9] != (abs(r[10]) <= args.dz_max)]
    for r in bad:
        print(f"  p{r[0]:02d} id{r[1]:<4d} area {r[2]:>6,} px   dv {r[6]:>+9,.0f} km/s"
              f"   n_z {r[7]:>3}   gal/star {r[4]/r[3]:.3f}   "
              f"area {'keep' if r[9] else 'drop'} / spec "
              f"{'keep' if abs(r[10]) <= args.dz_max else 'drop'}")
    if not bad:
        print("  (none)")

    # Threshold scan, so the criterion's sensitivity to it is visible. The equivalent
    # velocity depends on each pointing's own z, so the km/s column is a median.
    zm = np.median([r[5] - r[10] for r in rows])
    print(f"\n{'dz_max':>10} {'~km/s':>9} | {'keep':>6} | {'drop':>6} | dropped members")
    print("-" * 74)
    for v in (3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 0.3):
        drop = [f"p{r[0]:02d}id{r[1]}" for r in rows if abs(r[10]) > v]
        print(f"{v:>10.4g} {C_KMS * v / (1 + zm):>9,.0f} | "
              f"{len(rows)-len(drop):>6} | {len(drop):>6} | "
              f"{', '.join(drop) if drop else '(none)'}")


if __name__ == "__main__":
    main()
