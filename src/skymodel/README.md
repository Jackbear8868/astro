## Sex Extractro

### 1. Install
```bash
conda install -n astro -c conda-forge astromatic-source-extractor
conda activate sex
sex --version
```

### 2. What are files are

| Files | Character |
|---|---|
| `default.sex` | **Main Setting file**：detection threshold（1.0σ）、MINAREA（10）、BACK_SIZE（64）、Filter、Output settings |
| `default.param` | catalog 要輸出哪些欄位（NUMBER、X/Y_IMAGE、FLUX_ISO、ISOAREA、FLAGS、CLASS_STAR） |
| `default.conv` | ``all-ground'' convolution mask with FWHM = 2 pixels |
| `default.nnw` | 星系/恆星分類（CLASS_STAR）的類神經網路權重——照用即可 |

### 3. How to use Sex Extractor
```bash
cd src/skymodel/SExtractor
sex  <A 2D FITS image>  -c default.sex

sex ../../../results/skymodel/step01/whitelight.fits -c default.sex -CATALOG_NAME ../../../results/skymodel/step01/test.cat -CHECKIMAGE_NAME ../../../results/skymodel/step01/seg.fits,../../../results/skymodel/step01/nosky.fits
```
1. The image should be 2D and without Nan value.

