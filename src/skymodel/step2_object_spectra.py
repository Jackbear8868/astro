from pathlib import Path
import numpy as np
from astropy.io import fits


def sum_spectra_by_id(cube_path, seg, ids, chunk=200, var_path=None):
    """把屬於同一個 segmentation ID 的所有 spaxel 光譜加總。

    Parameters
    ----------
    cube_path : path-like
        MUSE cube;需要 DATA extension。
    var_path : path-like or None
        STAT(variance)從哪個檔案讀,預設和 cube_path 同一個。扣過天空的 cube
        只有 DATA,沒有 STAT —— 扣掉一個確定性的天空模型不改變像素的變異數,
        所以直接沿用原始 cube 的 STAT 是對的。
    seg : ndarray, shape (ny, nx)
        segmentation map,每格存所屬源的 ID,0 表示不屬於任何源。
        呼叫前必須先把視場外歸 0,否則視場外的像素會被一起加進來。
    ids : ndarray, shape (n_ids,)
        要處理的 ID 清單。
    chunk : int
        一次讀入幾個波長平面。只影響記憶體與速度,不改變結果。

    Returns
    -------
    flux : ndarray, shape (n_ids, nz)
        加總光譜。
    var : ndarray, shape (n_ids, nz)
        加總 variance。獨立像素相加時可加的是 variance 而不是 sigma,
        所以這裡直接相加;開根號之後才是加總光譜的雜訊。
    nspax : ndarray, shape (n_ids, nz)
        每個 ID 在每個波長「實際有資料」的 spaxel 數。cube 中存在壞 spaxel
        與波長邊界的缺值,所以這個數字會低於該 ID 的總 spaxel 數,而且隨波長
        變動。下游若要把加總換算成平均,除數必須用它而不是總 spaxel 數:

            mean_flux = flux / nspax
            mean_var  = var / nspax**2

        其中 variance 除的是 nspax 的平方,因為平均值的變異數是總和的 1/n²。

    Notes
    -----
    flux、var、nspax 三者計數的必定是同一批 spaxel:先用 ok 遮罩把不可用的
    位置清成 0,再直接相加。若改用 np.nansum 分別處理 flux 與 var,兩者會各自
    獨立跳過不同的位置(例如 flux 有值但 variance 是 NaN 的像素會進 flux 卻不
    進 var),使 variance 與 flux 不對應。
    """
    seg_flat = seg.ravel() # 2D -> 1D
    members  = [np.flatnonzero(seg_flat == i) for i in ids]

    with fits.open(cube_path, memmap=True) as hdul, \
         fits.open(var_path or cube_path, memmap=True) as vdul:
        nz   = hdul["DATA"].header["NAXIS3"]
        flux = np.zeros((len(ids), nz))
        var  = np.zeros((len(ids), nz))
        nspax = np.zeros((len(ids), nz))

        for j in range(0, nz, chunk):
            d = np.asarray(hdul["DATA"].data[j:j+chunk], np.float64).reshape(-1, seg_flat.size)
            v = np.asarray(vdul["STAT"].data[j:j+chunk], np.float64).reshape(-1, seg_flat.size)

            ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
            d  = np.where(ok, d, 0.0)
            v  = np.where(ok, v, 0.0)

            for k, idx in enumerate(members):
                flux[k,  j:j+chunk] = d[:, idx].sum(axis=1)
                var[k,   j:j+chunk] = v[:, idx].sum(axis=1)
                nspax[k, j:j+chunk] = ok[:, idx].sum(axis=1)

    return flux, var, nspax


def main():
    import argparse
    ap = argparse.ArgumentParser(description="按 segmentation ID 加總源光譜")
    ap.add_argument("--cube", required=True,
                    help="要萃取的 cube(取它的 DATA)。要含天光的原始 cube;"
                         "分類要用扣過天空的版本(step05 的 sky_subtracted.fits)")
    ap.add_argument("--var-cube", default=None,
                    help="STAT 從哪裡讀,預設同 --cube。我們自己扣天空的 cube 只有 "
                         "DATA,要用這個指到原始 cube")
    ap.add_argument("--work", required=True, help="這顆 cube 的工作區")
    ap.add_argument("--out", required=True, help="輸出目錄")
    args = ap.parse_args()
    work   = Path(args.work)
    STEP01 = work / "step01"
    out    = Path(args.out) if args.out else work / "step02"
    out.mkdir(parents=True, exist_ok=True)
    print(f"工作區 {work}   cube {Path(args.cube).name}")

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")

    valid_mask  = white != 0
    source_mask = (seg > 0) & valid_mask
    seg_valid   = np.where(valid_mask, seg, 0)      # 視場外一律歸 0,不參與加總

    ids, counts = np.unique(seg_valid[source_mask], return_counts=True)
    print(f"{len(ids)} sources, {counts.sum()} source spaxels")

    print(f"DATA <- {Path(args.cube).name}   STAT <- "
          f"{Path(args.var_cube or args.cube).name}")
    flux, var, nspax = sum_spectra_by_id(args.cube, seg_valid, ids,
                                         var_path=args.var_cube)

    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.nanmedian(flux / np.sqrt(var), axis=1)

    order = np.argsort(snr)[::-1]
    print(f"{'ID':>5} {'N':>7} {'sqrt(N)':>9} {'median SNR':>12}")
    for k in order[:20]:
        print(f"{ids[k]:>5d} {counts[k]:>7d} {np.sqrt(counts[k]):>9.1f} {snr[k]:>12.2f}")

    np.save(out / "object_ids.npy",   ids)
    np.save(out / "object_flux.npy",  flux)
    np.save(out / "object_var.npy",   var)
    np.save(out / "object_nspax.npy", nspax)
    print("saved ->", out)


if __name__ == "__main__":
    main()