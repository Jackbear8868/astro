from pathlib import Path
import numpy as np
from astropy.io import fits


def sum_spectra_by_id(cube_path, seg, ids, chunk=200):
    """把屬於同一個 segmentation ID 的所有 spaxel 光譜加總。

    Parameters
    ----------
    cube_path : path-like
        MUSE cube;需要 DATA(光譜)與 STAT(variance)兩個 extension。
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

    with fits.open(cube_path, memmap=True) as hdul:
        nz   = hdul["DATA"].header["NAXIS3"]
        flux = np.zeros((len(ids), nz))
        var  = np.zeros((len(ids), nz))
        nspax = np.zeros((len(ids), nz))

        for j in range(0, nz, chunk):
            d = np.asarray(hdul["DATA"].data[j:j+chunk], np.float64).reshape(-1, seg_flat.size)
            v = np.asarray(hdul["STAT"].data[j:j+chunk], np.float64).reshape(-1, seg_flat.size)

            ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
            d  = np.where(ok, d, 0.0)
            v  = np.where(ok, v, 0.0)

            for k, idx in enumerate(members):
                flux[k,  j:j+chunk] = d[:, idx].sum(axis=1)
                var[k,   j:j+chunk] = v[:, idx].sum(axis=1)
                nspax[k, j:j+chunk] = ok[:, idx].sum(axis=1)

    return flux, var, nspax

ROOT   = Path(__file__).resolve().parents[2]
STEP01 = ROOT / "results/skymodel/step01"
STEP02 = ROOT / "results/skymodel/step02"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

def main():
    STEP02.mkdir(parents=True, exist_ok=True)

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")

    valid_mask  = white != 0
    source_mask = (seg > 0) & valid_mask
    seg_valid   = np.where(valid_mask, seg, 0)      # 視場外一律歸 0,不參與加總

    ids, counts = np.unique(seg_valid[source_mask], return_counts=True)
    print(f"{len(ids)} sources, {counts.sum()} source spaxels")

    flux, var, nspax = sum_spectra_by_id(WSKY, seg_valid, ids)

    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.nanmedian(flux / np.sqrt(var), axis=1)

    order = np.argsort(snr)[::-1]
    print(f"{'ID':>5} {'N':>7} {'sqrt(N)':>9} {'median SNR':>12}")
    for k in order[:20]:
        print(f"{ids[k]:>5d} {counts[k]:>7d} {np.sqrt(counts[k]):>9.1f} {snr[k]:>12.2f}")

    np.save(STEP02 / "object_ids.npy",   ids)
    np.save(STEP02 / "object_flux.npy",  flux)
    np.save(STEP02 / "object_var.npy",   var)
    np.save(STEP02 / "object_nspax.npy", nspax)
    print("saved ->", STEP02)


if __name__ == "__main__":
    main()