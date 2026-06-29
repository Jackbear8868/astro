# 天空扣除論文庫（依策略分類）

依「預測 sky / 預測 residual / 混合 / 其他」分資料夾管理。分類依據與完整摘要見
[`../literature/sky-subtraction-papers.md`](../literature/sky-subtraction-papers.md)。
所有 PDF 由 arXiv 下載（公開取用）。

```
predict-sky/        策略一：建模天空本身再 data − sky
  empirical/        經驗/觀測法（sky fiber、nod-shuffle、Kelson、skycorr…）
  data-driven/      低秩/NMF 建天空
  ml/               機器學習建天空
  physical-model/   物理/合成天空模型（Cerro Paranal、PALACE…）
predict-residual/   策略二：先粗扣，再 PCA/資料驅動移除殘餘（ZAP 家族）
mixed/              pipeline 同時用兩策略（MUSE、DESI、SAMI、LAMOST）
other/
  sky-line-atlas/   天光發射線表/atlas（建模輸入）
  telluric/         telluric 吸收校正（姊妹問題）
  ml-spectral/      ML 光譜去噪/對殘餘 robust/分類
  strategy/         觀測策略比較
```

## 各檔對應 arXiv ID

### predict-sky/empirical
- Kelson2003_background-subtraction — `astro-ph/0303507`
- Glazebrook2001_nod-shuffle — `astro-ph/0011104`
- Davies2007_remove-residual-OH — `astro-ph/0612257`
- Noll2014_skycorr — `1405.3679`
- Law2016_MaNGA-DRP — `1607.08619`
- Sanchez2016_CALIFA-DR3 — `1604.02289`
- Rodrigues2010_sky-extraction-multifiber — `1009.0554`
- accurate-sky-continuum-fibre — `1302.3620`

### predict-sky/data-driven
- Kolganov2023_NMF-sky-subtraction — `2312.06761`

### predict-sky/ml
- Zhang2025_SMI-mutual-info-ML — `2508.19875`（唯一 DL 預測 sky）

### predict-sky/physical-model
- Noll2012_Cerro-Paranal-sky-model — `1205.2003`
- Jones2013_scattered-moonlight — `1310.7030`
- Noll2025_PALACE-airglow — `2504.10683`
- Patat2008_dancing-sky — `0801.2270`

### predict-residual
- KurtzMink2000_eigenvector-sky-subtraction — `astro-ph/0003112`
- WildHewett2005_OH-forest — `astro-ph/0501460`
- WildHewett2010_OH-forest-DR7 — `1010.2500`
- SharpParkinson2010_poisson-limit — `1007.0648`
- Soto2016_ZAP — `1602.08037` ★ MUSE/IFU 標準
- Marchetti2017_VIPERS-PCA — `1612.01825`
- Husemann2022_CARS-CubePCA — `2111.10417`
- Uzsoy2025_bayesian-component-separation — `2504.06870`

### mixed
- Weilbacher2020_MUSE-pipeline — `2006.08638`
- Guy2023_DESI-pipeline — `2209.14482`
- Croom2021_SAMI-DR3 — `2101.12224`
- Bai2017_LAMOST-sky-subtraction — `1705.02079`

### other/sky-line-atlas
- Oliva2015_GIANO-sky-lines — `1506.09004`
- Viuho2025_NIR-airglow-continuum — `2506.02102`

### other/telluric
- Smette2015_molecfit-I — `1501.07239`
- Kausch2015_molecfit-II — `1501.07237`
- Sedaghat2023_stellar-karaoke — `2301.00313`
- telluric2021_autoencoder-unmixing — `2111.09081`

### other/ml-spectral
- Melchior2023_SPENDER-I — `2211.07890`
- Melchior2023_SPENDER-II — `2302.02496`
- Camilleri2025_denoising-SDSS — `2510.08411`
- UNet2025_denoising-stellar — `2504.02523`
- Mukae2026_HETDEX-LAE-CNN — `2604.12414`
- MaNGA2026_anomaly-autoencoder — `2603.03734`
- CNN-transformer2026_denoiser — `2605.04434`
- superres2026_galaxy-spectra — `2603.18357`
- ViT2025_spectral-analysis — `2506.00294`
- DESI-DR2-QA2026 — `2606.21035`

### other/strategy
- Rodrigues2012_onsky-tests — `1609.06142`

---

## 無 arXiv、未下載（付費牆／會議論文）— 附 DOI/出處供自行取用

| 論文 | 類別 | 出處 |
|---|---|---|
| Streicher et al. 2011 — MUSE pipeline sky subtraction | predict-sky | ADS 2011ASPC..442..257S |
| Han, Song & Zhao 2023 — LAMOST bright night | predict-sky | MNRAS 526,5520 · DOI 10.1093/mnras/stad3115 |
| Zhang, Zhang & Ye 2016 — NMF+sparsity sky | predict-sky | PASA 33,e058 |
| Krisciunas & Schaefer 1991 — moonlight brightness | predict-sky(物理) | PASP 103,1033 · DOI 10.1086/132921 |
| Yoachim et al. 2016 — LSST sky brightness model | predict-sky(物理) | SPIE 9910 |
| Hart 2019 — Sky Residual Correction | predict-residual | AJ 157,213 · DOI 10.3847/1538-3881/ab1a35 |
| Hanuschik 2003 — UVES sky emission atlas | other/atlas | A&A 407,1157 |
| Rousselot et al. 2000 — NIR OH line list | other/atlas | A&A 354,1134 |
| Cosby et al. 2006 — nightglow atlas | other/atlas | JGR 111,A12307 · DOI 10.1029/2006JA012023 |
| ESO SkyCalc（工具，無論文）| predict-sky(物理) | 引用 Noll 2012 / Jones 2013 |
| Subaru PFS sky subtraction (2024)（待驗證）| predict-sky | SPIE 13096,130962M · DOI 10.1117/12.3015628 |
</content>
